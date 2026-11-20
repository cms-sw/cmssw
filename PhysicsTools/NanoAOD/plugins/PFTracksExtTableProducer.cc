#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MuonAnalysis/MuonAssociators/interface/PropagateToMuonSetup.h"

class PFTracksExtTableProducer : public edm::global::EDProducer<> {
public:
  explicit PFTracksExtTableProducer(const edm::ParameterSet& iConfig)
      : src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"))),
        name_(iConfig.getParameter<std::string>("name")),
        propSetup_(iConfig.getParameter<edm::ParameterSet>("propagation"), consumesCollector()) {
    produces<nanoaod::FlatTable>();
  }
  ~PFTracksExtTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("pfTracks"));
    desc.add<std::string>("name", "Track");
    edm::ParameterSetDescription propDesc;
    PropagateToMuonSetup::fillPSetDescription(propDesc);
    desc.add<edm::ParameterSetDescription>("propagation", propDesc);
    descriptions.add("pfTracksExtTable", desc);
  }

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<reco::TrackCollection> src_;
  const std::string name_;
  const PropagateToMuonSetup propSetup_;
};

void PFTracksExtTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& tracks = iEvent.get(src_);
  const PropagateToMuon propagator = propSetup_.init(iSetup);

  const size_t ntrk = tracks.size();
  std::vector<float> propEta(ntrk, -99.), propPhi(ntrk, -99.), propR(ntrk, -99.);

  for (size_t i = 0; i < ntrk; ++i) {
    const TrajectoryStateOnSurface tsos = propagator.extrapolate(tracks[i]);
    if (!tsos.isValid())
      continue;
    const GlobalPoint pos = tsos.globalPosition();
    propEta[i] = pos.eta();
    propPhi[i] = pos.phi();
    propR[i] = pos.perp();
  }

  auto table = std::make_unique<nanoaod::FlatTable>(ntrk, name_, false, true);
  table->addColumn<float>("propEta", propEta, "eta of the track extrapolated to muon station 1 (-99 if failed)", 10);
  table->addColumn<float>("propPhi", propPhi, "phi of the track extrapolated to muon station 1 (-99 if failed)", 10);
  table->addColumn<float>(
      "propR", propR, "transverse radius (cm) of the track extrapolated to muon station 1 (-99 if failed)", 10);

  iEvent.put(std::move(table));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFTracksExtTableProducer);
