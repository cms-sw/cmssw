#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class GEDGsfElectronValueMapProducer : public edm::global::EDProducer<> {
public:
  explicit GEDGsfElectronValueMapProducer(const edm::ParameterSet&);
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::GsfElectronCollection> electronsToken_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfCandsToken_;
  const edm::EDPutTokenT<edm::ValueMap<reco::GsfElectronRef>> putToken_;
};

GEDGsfElectronValueMapProducer::GEDGsfElectronValueMapProducer(const edm::ParameterSet& cfg)
    : electronsToken_(consumes<reco::GsfElectronCollection>(cfg.getParameter<edm::InputTag>("gedGsfElectrons"))),
      pfCandsToken_(consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("egmPFCandidatesTag"))),
      putToken_{produces<edm::ValueMap<reco::GsfElectronRef>>()} {}

void GEDGsfElectronValueMapProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup&) const {
  auto electrons = event.getHandle(electronsToken_);
  auto pfCandidatesHandle = event.getHandle(pfCandsToken_);

  // ValueMap
  edm::ValueMap<reco::GsfElectronRef> valMap{};
  edm::ValueMap<reco::GsfElectronRef>::Filler valMapFiller(valMap);

  //Loop over the collection of PFFCandidates
  std::vector<reco::GsfElectronRef> values;

  for (auto const& pfCandidate : *pfCandidatesHandle) {
    reco::GsfElectronRef myRef;
    // First check that the GsfTrack is non null
    if (pfCandidate.gsfTrackRef().isNonnull()) {
      // now look for the corresponding GsfElectron
      const auto itcheck = std::find_if(electrons->begin(), electrons->end(), [&pfCandidate](const auto& ele) {
        return (ele.gsfTrack() == pfCandidate.gsfTrackRef());
      });
      if (itcheck != electrons->end()) {
        // Build the Ref from the handle and the index
        myRef = reco::GsfElectronRef(electrons, itcheck - electrons->begin());
      }
    }
    values.push_back(myRef);
  }
  valMapFiller.insert(pfCandidatesHandle, values.begin(), values.end());

  valMapFiller.fill();
  event.emplace(putToken_, valMap);
}

void GEDGsfElectronValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gedGsfElectrons", {"gedGsfElectronsTmp"});
  desc.add<edm::InputTag>("egmPFCandidatesTag", {"particleFlowEGamma"});
  descriptions.add("gedGsfElectronValueMapsTmp", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEDGsfElectronValueMapProducer);
