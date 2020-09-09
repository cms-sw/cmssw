
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/ME0Muon.h"
#include "DataFormats/MuonReco/interface/ME0MuonCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoMuon/MuonIdentification/interface/ME0MuonSelector.h"

#include <memory>

class ME0MuonTrackCollProducer : public edm::stream::EDProducer<> {
public:
  explicit ME0MuonTrackCollProducer(const edm::ParameterSet&);
  ~ME0MuonTrackCollProducer() override = default;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  const edm::ParameterSet parset_;
  edm::EDGetTokenT<ME0MuonCollection> OurMuonsToken_;
};

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ME0MuonTrackCollProducer);

ME0MuonTrackCollProducer::ME0MuonTrackCollProducer(const edm::ParameterSet& parset)
    : parset_(parset), OurMuonsToken_(consumes<ME0MuonCollection>(parset.getParameter<edm::InputTag>("me0MuonTag"))) {
  produces<reco::TrackCollection>();
}

void ME0MuonTrackCollProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace reco;
  using namespace edm;
  Handle<ME0MuonCollection> OurMuons;
  iEvent.getByToken(OurMuonsToken_, OurMuons);

  std::unique_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);

  reco::TrackRefProd rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();

  for (std::vector<reco::ME0Muon>::const_iterator thismuon = OurMuons->begin(); thismuon != OurMuons->end();
       ++thismuon) {
    if (!muon::me0::isGoodMuon(*thismuon, muon::me0::Tight))
      continue;
    reco::TrackRef trackref;

    if (thismuon->innerTrack().isNonnull())
      trackref = thismuon->innerTrack();

    const reco::Track* trk = &(*trackref);
    selectedTracks->push_back(*trk);
  }
  iEvent.put(std::move(selectedTracks));
}
