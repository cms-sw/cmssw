#include "Calibration/EcalAlCaRecoProducers/plugins/AlCaElectronTracksReducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

//#define ALLrecHits
//#define DEBUG

//#define QUICK -> if commented loop over the recHits of the SC and add them to the list of recHits to be saved
//                 comment it if you want a faster module but be sure the window is large enough

AlCaElectronTracksReducer::AlCaElectronTracksReducer(const edm::ParameterSet& iConfig) 
{

  //gsfTracksLabel_ = iConfig.getParameter< edm::InputTag > ("gsfTracksLabel");
  generalTracksLabel_ = iConfig.getParameter< edm::InputTag > ("generalTracksLabel");
  generalTracksExtraLabel_ = iConfig.getParameter< edm::InputTag > ("generalTracksExtraLabel");
  electronLabel_ = iConfig.getParameter< edm::InputTag > ("electronLabel");

  // name of the output collection
  //gsfTracksCollection_ = iConfig.getParameter<std::string>("gsfTracksCollection");
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

  // get the ECAL geometry:
//   ESHandle<CaloGeometry> geoHandle;
//   iSetup.get<CaloGeometryRecord>().get(geoHandle);

//   edm::ESHandle<CaloTopology> theCaloTopology;
//   iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
//   const CaloTopology *caloTopology = theCaloTopology.product();
  
  // Get GSFElectrons
  Handle<reco::GsfElectronCollection> pElectrons;
  iEvent.getByLabel(electronLabel_, pElectrons);
//   if (!pElectrons.isValid()) {
//     edm::LogError ("reading") << electronLabel_ << " not found" ; 
//     return ;
//   }
  
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
#ifndef CMSSW42X
    TrackRef track = (eleIt-> closestTrack() ) ;
#else
    TrackRef track = (eleIt-> closestCtfTrackRef());
#endif
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

