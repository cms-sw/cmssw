// -*- C++ -*-
//
// Package:    TrackHitFilter
// Class:      TrackHitFilter
// 
/**\class TrackHitFilter TrackHitFilter.cc Alignment/TrackHitFilter/src/TrackHitFilter.cc

 Description: Selects some track hits for refitting input 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Roberto Covarelli
//         Created:  Mon Jan 15 10:39:42 CET 2007
// $Id: TrackHitFilter.cc,v 1.12 2009/02/17 15:00:31 castello Exp $
//
//

#include "Alignment/TrackHitFilter/interface/TrackHitFilter.h"
#include "FWCore/Framework/interface/Event.h" 
#include "FWCore/Framework/interface/MakerMacros.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h" 
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/TrackReco/interface/TrackExtra.h" 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h" 
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h" 
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"

#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

// RC
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"




using namespace edm;
using namespace reco;

//
// constructors and destructor
//
TrackHitFilter::TrackHitFilter(const edm::ParameterSet& iConfig):
  theSrc( iConfig.getParameter<edm::InputTag>( "src" ) ),
  theHitSel( iConfig.getParameter<std::string>( "hitSelection" ) ),
  theMinHits( iConfig.getParameter<unsigned int>( "minHitsForRefit" ) ),
  rejectBadMods( iConfig.getParameter<bool>( "rejectBadMods" ) ),
  theBadMods( iConfig.getParameter<std::vector<unsigned int> >( "theBadModules" ) ),
  rejectBadStoNHits( iConfig.getParameter<bool>( "rejectBadStoNHits" ) ),
  theCMNSubtractionMode(iConfig.getUntrackedParameter<std::string>( "CMNSubtractionMode" ,"Median") ) ,
  theStoNthreshold( iConfig.getParameter<double>( "theStoNthreshold" ) ),  
  rejectBadClusterPixelHits( iConfig.getParameter<bool>( "rejectBadClusterPixelHits" ) ),
  thePixelClusterthreshold( iConfig.getParameter<double>( "thePixelClusterthreshold" ) )
{

   //register your products, and/or set an "alias" label
  produces<TrackCollection>();
  produces<TrackExtraCollection>();
  produces<TrackingRecHitCollection>();
 
}


TrackHitFilter::~TrackHitFilter()
{
}

void TrackHitFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   //Read track collection from the Event
   Handle<TrackCollection> trackAllHits;
   iEvent.getByLabel(theSrc,trackAllHits);

   // Create empty Track, TrackExtra and TrackingRecHits collections
   std::auto_ptr<TrackCollection> trackSelectedHits( new TrackCollection() );
   std::auto_ptr<TrackExtraCollection> txSelectedHits( new TrackExtraCollection() );
   std::auto_ptr<TrackingRecHitCollection> trhSelectedHits( new TrackingRecHitCollection() );

   //get references
   TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
   TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<TrackExtraCollection>();
   TrackRefProd rTracks = iEvent.getRefBeforePut<TrackCollection>();
   edm::Ref<TrackExtraCollection>::key_type idx = 0;
   edm::Ref<TrackExtraCollection>::key_type hidx = 0;

   TrackerAlignableId* TkMap = new TrackerAlignableId();
   
   LogDebug("HitFilter")<< trackAllHits->size() << " track(s) found in the event with label " << theSrc<<std::endl;

   unsigned int nTr = 0;
   std::vector<unsigned int> accHits;
   std::vector<unsigned int> allHits;

   // first iteration : count hits
   for( TrackCollection::const_iterator iTrack = trackAllHits->begin(); iTrack != trackAllHits->end(); ++iTrack ) {
       
     const Track * trk = &(*iTrack);
     
     unsigned int allhits = 0;
     unsigned int acchits = 0;
     for (trackingRecHit_iterator iHit = trk->recHitsBegin(); iHit != trk->recHitsEnd(); iHit++) {
       
       allhits++;
       TrackingRecHit* hit = (*iHit)->clone();
       //std::cout<<"See if it crashes. Geographical Id of the hit: "<<hit->geographicalId()<<std::endl;
       std::pair<int,int> typeAndLay = TkMap->typeAndLayerFromDetId( hit->geographicalId() );
       int type = typeAndLay.first;   
       int layer = typeAndLay.second;
       DetId detid= hit->geographicalId();
       if (keepThisHit( detid, type, layer, hit, iEvent,iSetup )) acchits++; 
       
     }//end loop on hits
     
     if (!nTr) {
       LogDebug("HitFilter") << "TrackHitFilter **** In first track " << acchits << " RecHits retained out of " << allhits;
       nTr++;
     }
     allHits.push_back(allhits);
     accHits.push_back(acchits);
     
   }//end loop on tracks

   nTr = 0;
   // second iteration : store tracks
   for( TrackCollection::const_iterator iTrack2 = trackAllHits->begin(); iTrack2 != trackAllHits->end(); ++iTrack2 ) {
       
     if (accHits.at(nTr) >= theMinHits) {
       const Track * trk = &(*iTrack2);
       Track * myTrk = new Track(*trk);
       PropagationDirection seedDir = trk->seedDirection();
       myTrk->setExtra( TrackExtraRef( rTrackExtras, idx ++ ) );
       TrackExtra * tx = new TrackExtra( trk->outerPosition(), trk->outerMomentum(), 
					   trk->outerOk(), trk->innerPosition(), 
					   trk->innerMomentum(), trk->innerOk(),
					   trk->outerStateCovariance(), trk->outerDetId(),
					   trk->innerStateCovariance(), trk->innerDetId() , seedDir ) ;	

       unsigned int i = 0; 
       for (trackingRecHit_iterator iHit = trk->recHitsBegin(); iHit != trk->recHitsEnd(); iHit++) {

	 TrackingRecHit* hit = (*iHit)->clone();
	 std::pair<int,int> typeAndLay = TkMap->typeAndLayerFromDetId( hit->geographicalId() );
	 int type = typeAndLay.first;   
	 int layer = typeAndLay.second;
	 
	 if (keepThisHit( hit->geographicalId(), type, layer, hit, iEvent, iSetup )) {
   
	   myTrk->setHitPattern( * hit, i ++ );
	   trhSelectedHits->push_back( hit );
	   tx->add( TrackingRecHitRef( rHits, hidx ++ ) );

	 }
       } 
       
       trackSelectedHits->push_back( *myTrk );
       txSelectedHits->push_back( *tx );
     }
       
     nTr++;
   }
    
   iEvent.put( trackSelectedHits );     
   iEvent.put( txSelectedHits );
   iEvent.put( trhSelectedHits );
   //   std::cout<<"Finished this call to <TrackHitFilter::produce>"<<std::endl;
}


void 
TrackHitFilter::beginJob(const edm::EventSetup&)
{
}

void 
TrackHitFilter::endJob() {
}

bool TrackHitFilter::keepThisHit(DetId id, int type, int layer, const TrackingRecHit* therechit, const edm::Event &iEvent, const edm::EventSetup& iSetup) 
{
  bool keepthishit = true;
  
  if ( theHitSel == "PixelOnly" ) {
    if (abs(type)<1 || abs(type)>2) keepthishit = false; 
  }
  else if ( theHitSel == "PixelBarrelOnly" ) {
    if (abs(type)!=1) keepthishit = false; 
  }
  else if ( theHitSel == "PixelAndDSStripBarrelOnly" ) {
    if (!((abs(type)==1)
	  || ((abs(type)==3 || abs(type)==5) && layer<2))) keepthishit = false;
  }	   
  else if ( theHitSel == "SiStripOnly" ) {
    if (abs(type)>=1 && abs(type)<=2) keepthishit = false; 
  } 
  else if ( theHitSel == "TOBOnly" ) {
     if (abs(type)!=5) keepthishit = false;
  }
  else if ( theHitSel == "TOBandTIBOnly" ) {
     if ( !(abs(type)==3 || abs(type)==5) )  keepthishit = false;
  }
  else if ( theHitSel == "TOBandTIBl4Only" ) {
    if (!(abs(type)==5 || (abs(type)==3 && layer>=4))) keepthishit = false;
  }
  else if ( theHitSel == "TOBandTIBl34Only" ) {
    if (!(abs(type)==5 || (abs(type)==3 && layer>=3))) keepthishit = false;
  }
  else if ( theHitSel == "TOBandTIBl234Only" ) {
    if (!(abs(type)==5 || (abs(type)==3 && layer>=2))) keepthishit = false;
  }
  else if ( theHitSel == "noPXE" ) {
    if (abs(type)==2) keepthishit = false;
  }
  
  if (rejectBadMods) {
    for( std::vector<unsigned int>::const_iterator iMod = theBadMods.begin(); iMod != theBadMods.end(); ++iMod ) {  
      if (id.rawId() == *iMod) {
	keepthishit = false;
        LogDebug("HitFilter") << "TrackHitFilter **** Rejected a hit on module " << id.rawId();
      }
    }
  }


  //********* Rejects pixel hits with 'bad' cluster 

  if(rejectBadClusterPixelHits && (abs(type)==1 || abs(type)==2)){

    const SiPixelRecHit* pixelhit = dynamic_cast<const SiPixelRecHit*>(therechit);
    const SiPixelCluster* pixelcluster = &*(pixelhit->cluster());
    if(pixelcluster->size()==1 && pixelcluster->charge()<thePixelClusterthreshold){keepthishit = false;}
  }


  // ****************        
  // Reject hits with bad S/N
  if (rejectBadStoNHits && (abs(type)>2) ) { //apply it only to Strip hits
    /*** RC ****/ 
    //   const uint32_t& recHitDetId = id.rawId();
    const SiStripMatchedRecHit2D* matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(therechit);
    const SiStripRecHit2D* hit = dynamic_cast<const SiStripRecHit2D*>(therechit);
    const ProjectedSiStripRecHit2D* unmatchedhit = dynamic_cast<const ProjectedSiStripRecHit2D*>(therechit);

    if (matchedhit) {  
      bool keepmonohit=true;
      bool keepstereohit=true;

      const SiStripRecHit2D* monohit=matchedhit->monoHit();    
      const SiStripCluster* monocluster = &*(monohit->cluster());
      SiStripClusterInfo monoclusterInfo = SiStripClusterInfo( *monocluster, iSetup); 
      if (monoclusterInfo.signalOverNoise() < theStoNthreshold ) keepmonohit = false;
	

      const SiStripRecHit2D* stereohit=matchedhit->stereoHit();   
      const SiStripCluster* stereocluster = &*(stereohit->cluster());
      SiStripClusterInfo stereoclusterInfo = SiStripClusterInfo(*stereocluster, iSetup);   
      if (stereoclusterInfo.signalOverNoise() < theStoNthreshold )keepstereohit = false;
           
      if (!keepmonohit || !keepstereohit) keepthishit = false;    
    }
    else if (hit) {
      const SiStripCluster* cluster = &*(hit->cluster());
      SiStripClusterInfo clusterInfo = SiStripClusterInfo(*cluster, iSetup);     
       if (clusterInfo.signalOverNoise() < theStoNthreshold )keepthishit = false;
     
    }
    else if (unmatchedhit) {
      const SiStripRecHit2D &orighit = unmatchedhit->originalHit(); 
      const SiStripCluster* origcluster = &*(orighit.cluster());
      SiStripClusterInfo clusterInfo = SiStripClusterInfo(*origcluster, iSetup);     

      if (clusterInfo.signalOverNoise() < theStoNthreshold ) keepthishit = false;   
         
    }
  } // end reject bad S/N

  
  return keepthishit;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackHitFilter);
