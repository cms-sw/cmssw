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
// $Id$
//
//

#include "Alignment/TrackHitFilter/interface/TrackHitFilter.h"

using namespace edm;
using namespace reco;

//
// constructors and destructor
//
TrackHitFilter::TrackHitFilter(const edm::ParameterSet& iConfig):
  theSrc( iConfig.getParameter<edm::InputTag>( "src" ) ),
  theHitSel( iConfig.getParameter<std::string>( "hitSelection" ) )
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
   // iEvent.getByType(trackAllHits);

   // Create empty Track, TrackExtra and TrackingRecHits collections
   std::auto_ptr<TrackCollection> trackSelectedHits( new TrackCollection() );
   std::auto_ptr<TrackExtraCollection> txSelectedHits( new TrackExtraCollection() );
   std::auto_ptr<TrackingRecHitCollection> trhSelectedHits( new TrackingRecHitCollection() );

   TrackingRecHitRefProd rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
   TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<TrackExtraCollection>();
   TrackRefProd rTracks = iEvent.getRefBeforePut<TrackCollection>();
   edm::Ref<TrackExtraCollection>::key_type idx = 0;
   edm::Ref<TrackExtraCollection>::key_type hidx = 0;

   TrackerAlignableId* TkMap = new TrackerAlignableId();
   
   LogDebug("HitFilter") << trackAllHits->size() << 
     " track(s) found in the event with label " << theSrc;

   unsigned int nTr = 0;
   if (trackAllHits->size() > 0) { 
     for( TrackCollection::const_iterator iTrack = trackAllHits->begin(); iTrack != trackAllHits->end(); ++iTrack ) {
       
       try {
	 const Track * trk = &(*iTrack);
         Track * myTrk = new Track(*trk);
	 myTrk->setExtra( TrackExtraRef( rTrackExtras, idx ++ ) );
         TrackExtra * tx = new TrackExtra( trk->outerPosition(), trk->outerMomentum(), 
					   trk->outerOk(), trk->innerPosition(), 
					   trk->innerMomentum(), trk->innerOk(),
					   trk->outerStateCovariance(), trk->outerDetId(),
					   trk->innerStateCovariance(), trk->innerDetId() ) ;	 
	 unsigned int accHits = 0 ;
         unsigned int allHits = 0;
	 for (trackingRecHit_iterator iHit = trk->recHitsBegin(); iHit != trk->recHitsEnd(); iHit++) {

           allHits++;
	   TrackingRecHit * hit = (*iHit)->clone();
	   std::pair<int,int> typeAndLay = TkMap->typeAndLayerFromDetId( hit->geographicalId() );
	   int type = typeAndLay.first;   
	   int layer = typeAndLay.second;
	   bool keepThisHit = true;
	   
	   if ( theHitSel == "PixelOnly" ) {
	     if (abs(type)<1 || abs(type)>2) keepThisHit = false; 
	   }
	   else if ( theHitSel == "PixelBarrelOnly" ) {
	     if (abs(type)!=1) keepThisHit = false; 
	   }
	   else if ( theHitSel == "PixelAndDSStripBarrelOnly" ) {
	     if (!((abs(type)==1)
		   || ((abs(type)==3 || abs(type)==5) && layer<2))) keepThisHit = false;
	   }	   
           else if ( theHitSel == "SiStripOnly" ) {
	     if (abs(type)>=1 && abs(type)<=2) keepThisHit = false; 
	   } 

	   if (keepThisHit) {
	    
	     myTrk->setHitPattern( * hit, accHits ++ );
	     trhSelectedHits->push_back( hit );
	     tx->add( TrackingRecHitRef( rHits, hidx ++ ) );
	     
	   }

           delete hit;
	 } 
	
         // Set a minimum number of hits for KFrefitting
         if (accHits >= 2) {
	   trackSelectedHits->push_back( *myTrk );
	   txSelectedHits->push_back( *tx );
	 }

	 if (!nTr) LogDebug("HitFilter") << "TrackHitFilter **** In first track " << accHits << " RecHits retained out of " << allHits;
         nTr++;

       } catch (cms::Exception &e){}
       
     }
   
   }

     iEvent.put( trackSelectedHits );     
     iEvent.put( txSelectedHits );
     iEvent.put( trhSelectedHits );

}


void 
TrackHitFilter::beginJob(const edm::EventSetup&)
{
}

void 
TrackHitFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackHitFilter);
