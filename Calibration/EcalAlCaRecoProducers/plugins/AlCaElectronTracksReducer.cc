#include "Calibration/EcalAlCaRecoProducers/plugins/AlCaElectronTracksReducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

AlCaElectronTracksReducer::AlCaElectronTracksReducer(const edm::ParameterSet& iConfig) {
  generalTracksToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("generalTracksLabel"));
  generalTracksExtraToken_ =
      consumes<reco::TrackExtraCollection>(iConfig.getParameter<edm::InputTag>("generalTracksExtraLabel"));
  electronToken_ = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electronLabel"));

  // name of the output collection
  alcaTrackExtraCollection_ = iConfig.getParameter<std::string>("alcaTrackExtraCollection");

  //register your products
  produces<reco::TrackCollection>(alcaTrackCollection_);
  produces<reco::TrackExtraCollection>(alcaTrackExtraCollection_);
}

// ------------ method called to produce the data  ------------
void AlCaElectronTracksReducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  using namespace std;
  using namespace reco;

  // Get GSFElectrons
  Handle<reco::GsfElectronCollection> pElectrons;
  iEvent.getByToken(electronToken_, pElectrons);

  const reco::GsfElectronCollection* electronCollection = pElectrons.product();

  Handle<TrackCollection> generalTracksHandle;
  iEvent.getByToken(generalTracksToken_, generalTracksHandle);

  Handle<TrackExtraCollection> generalTracksExtraHandle;
  iEvent.getByToken(generalTracksExtraToken_, generalTracksExtraHandle);

  //Create empty output collections
  auto redGeneralTracksCollection = std::make_unique<TrackCollection>();
  auto redGeneralTracksExtraCollection = std::make_unique<TrackExtraCollection>();

  reco::GsfElectronCollection::const_iterator eleIt;

  for (eleIt = electronCollection->begin(); eleIt != electronCollection->end(); eleIt++) {
    // barrel
    TrackRef track = (eleIt->closestCtfTrackRef());
    if (track.isNull()) {
      //      edm::LogError("track") << "Track Ref not found " << eleIt->energy() << "\t" << eleIt->eta();
      continue;
    }
    redGeneralTracksCollection->push_back(*track);
    if (generalTracksExtraHandle.isValid())
      redGeneralTracksExtraCollection->push_back(*(track->extra()));
  }

  //Put selected information in the event
  iEvent.put(std::move(redGeneralTracksCollection), alcaTrackCollection_);
  iEvent.put(std::move(redGeneralTracksExtraCollection), alcaTrackExtraCollection_);
}

DEFINE_FWK_MODULE(AlCaElectronTracksReducer);
