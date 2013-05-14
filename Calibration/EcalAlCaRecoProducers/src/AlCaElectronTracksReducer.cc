#include "Calibration/EcalAlCaRecoProducers/interface/AlCaElectronTracksReducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


AlCaElectronTracksReducer::AlCaElectronTracksReducer(const edm::ParameterSet& iConfig) 
{

  generalTracksLabel_ = iConfig.getParameter< edm::InputTag > ("generalTracksLabel");
  generalTracksExtraLabel_ = iConfig.getParameter< edm::InputTag > ("generalTracksExtraLabel");
  electronLabel_ = iConfig.getParameter< edm::InputTag > ("electronLabel");

  // name of the output collection
  alcaTrackCollection_      = iConfig.getParameter<std::string>("alcaTrackCollection");
  alcaTrackExtraCollection_ = iConfig.getParameter<std::string>("alcaTrackExtraCollection");
  
  //register your products
  produces< reco::TrackCollection > (alcaTrackCollection_) ;
  produces< reco::TrackExtraCollection > (alcaTrackExtraCollection_) ;

}


AlCaElectronTracksReducer::~AlCaElectronTracksReducer()
{}


// ------------ method called to produce the data  ------------
void AlCaElectronTracksReducer::produce (edm::Event& iEvent, 
                                const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  using namespace reco;

  Handle<reco::GsfElectronCollection> pElectrons;
  iEvent.getByLabel(electronLabel_, pElectrons);
  
  const reco::GsfElectronCollection * electronCollection = 
    pElectrons.product();
  
  Handle<TrackCollection> generalTracksHandle;
  iEvent.getByLabel(generalTracksLabel_,generalTracksHandle);
  
  Handle<TrackExtraCollection> generalTracksExtraHandle;
  iEvent.getByLabel(generalTracksExtraLabel_,generalTracksExtraHandle);
 
  //Create empty output collections
  std::auto_ptr< TrackCollection > redGeneralTracksCollection (new TrackCollection) ;
  std::auto_ptr< TrackExtraCollection > redGeneralTracksExtraCollection (new TrackExtraCollection) ;
  
  reco::GsfElectronCollection::const_iterator eleIt;

  for (eleIt=electronCollection->begin(); eleIt!=electronCollection->end(); eleIt++) {
    // barrel
    TrackRef track = (eleIt-> closestTrack() ) ;
    if(track.isNull()){
      //      edm::LogError("track") << "Track Ref not found " << eleIt->energy() << "\t" << eleIt->eta();
      continue;
    }
    redGeneralTracksCollection->push_back(*track);
    if(generalTracksExtraHandle.isValid()) redGeneralTracksExtraCollection->push_back(*(track->extra()));
  }
  
  //Put selected information in the event
  iEvent.put( redGeneralTracksCollection, alcaTrackCollection_ );
  iEvent.put( redGeneralTracksExtraCollection, alcaTrackExtraCollection_ );
}


DEFINE_FWK_MODULE(AlCaElectronTracksReducer);

