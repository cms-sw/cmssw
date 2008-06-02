#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "RecoLocalTracker/SubCollectionProducers/interface/RemainingClusterProducer.h"
//
// class decleration
//
using namespace std;
using namespace edm;



RemainingClusterProducer::RemainingClusterProducer(const ParameterSet& iConfig):
  conf_(iConfig)
{
  produces< DetSetVector<SiPixelCluster> >();
  produces< DetSetVector<SiStripCluster> > ();
  

}


RemainingClusterProducer::~RemainingClusterProducer()
{
 


}



void
RemainingClusterProducer::produce(Event& iEvent, const EventSetup& iSetup)
{


  using namespace edm;
  InputTag rphirecHitsTag    = conf_.getParameter<InputTag>("rphirecHits");
  InputTag stereorecHitsTag  = conf_.getParameter<InputTag>("stereorecHits"); 
  InputTag pixelTag          = conf_.getParameter<InputTag>("pixelHits");
  InputTag tkTag             = conf_.getParameter<InputTag>("recTracks");  
  InputTag matchedrecHitsTag = conf_.getParameter<InputTag>("matchedRecHits");

  //HANDLE
  Handle<SiStripMatchedRecHit2DCollection> matchedrecHits; 
  Handle<SiStripRecHit2DCollection> rphirecHits;
  Handle<SiStripRecHit2DCollection> stereorecHits;
  Handle<SiPixelRecHitCollection> pixelHits;
  Handle<TrackingRecHitCollection> trackingHits;

  //GET COLLECTIONS
  iEvent.getByLabel( matchedrecHitsTag , matchedrecHits);
  iEvent.getByLabel( rphirecHitsTag    , rphirecHits);
  iEvent.getByLabel( stereorecHitsTag  , stereorecHits);
  iEvent.getByLabel( pixelTag          , pixelHits);
  iEvent.getByLabel( tkTag             , trackingHits);
  
  //
  thePixelClusterVector.clear();
  theStripClusterVector.clear();



  //DETSETVECTOR OF TRACKING HITS
  DetSetVector<const SiPixelRecHit *> pixelInTrack;
  DetSetVector<const SiStripRecHit2D*> monoInTrack;
  DetSetVector<const SiStripRecHit2D*> stereoInTrack;

  for(TrackerGeometry::DetContainer::const_iterator it = 
	pDD->dets().begin(); it != pDD->dets().end(); it++){
    DetId detid = ((*it)->geographicalId());
    unsigned int isub=detid.subdetId();
    //PIXEL
    if  ((isub==  PixelSubdetector::PixelBarrel) || (isub== PixelSubdetector::PixelEndcap)) {
      SiPixelRecHitCollection::range pixelrechitRange = (pixelHits.product())->get((detid));
      if (pixelrechitRange.second-pixelrechitRange.first!=0){
 	DetSet<const SiPixelRecHit *> Pix_tk(detid.rawId());
 	pixelInTrack.insert(Pix_tk);
      }
    }

    //STRIP 
    if ((uint(detid.subdetId())==StripSubdetector::TIB) ||
	(uint(detid.subdetId())==StripSubdetector::TOB) ||
	(uint(detid.subdetId())==StripSubdetector::TID) ||
	(uint(detid.subdetId())==StripSubdetector::TEC)){
      //rphi
      SiStripRecHit2DCollection::range monorechitRange = (rphirecHits.product())->get((detid));
      if (monorechitRange.second-monorechitRange.first!=0){
 	DetSet<const  SiStripRecHit2D*> Mono_tk(detid.rawId());
 	monoInTrack.insert(Mono_tk);
      }
      //stereo
      SiStripRecHit2DCollection::range stereorechitRange = (stereorecHits.product())->get((detid));
      if (stereorechitRange.second-stereorechitRange.first!=0){
 	DetSet<const  SiStripRecHit2D*> Stereo_tk(detid.rawId());
 	stereoInTrack.insert(Stereo_tk);
      }
    }
  }

  //FILL THE DETSETVECTOR
  TrackingRecHitCollection::const_iterator hit;
  for(hit=trackingHits.product()->begin();hit!=trackingHits.product()->end();hit++){
    if ((*hit).isValid()){
      const SiPixelRecHit* pixelhit=dynamic_cast<const SiPixelRecHit*>(&(*hit));
      if (pixelhit!=0)
	(*pixelInTrack.find(pixelhit->geographicalId().rawId())).push_back(pixelhit);

     

      const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*hit));
      if (matchedhit!=0) {
	(*stereoInTrack.find(matchedhit->stereoHit()->geographicalId().rawId())).
	  push_back(matchedhit->stereoHit());
	(*monoInTrack.find(matchedhit->monoHit()->geographicalId().rawId())).
	  push_back(matchedhit->monoHit());
      }

      const SiStripRecHit2D* monohit=dynamic_cast<const SiStripRecHit2D*>(&(*hit)); 
      if (monohit!=0)
	(*monoInTrack.find(monohit->geographicalId().rawId())).push_back(monohit);
    }
  }
  vector<const SiPixelRecHit*>::const_iterator ipix;
  vector<const SiStripRecHit2D*>::const_iterator istrip;
 

  //LOOP OVER ALL THE HITS
  for(TrackerGeometry::DetContainer::const_iterator it = 
	pDD->dets().begin(); it != pDD->dets().end(); it++){

    DetId detid = ((*it)->geographicalId());
    unsigned int isub=detid.subdetId();
    
    //PIXEL
    if  ((isub==  PixelSubdetector::PixelBarrel) || (isub== PixelSubdetector::PixelEndcap)) {  
      
      SiPixelRecHitCollection::range pixelrechitRange = (pixelHits.product())->get((detid));
      SiPixelRecHitCollection::const_iterator pixeliter =pixelrechitRange.first;
      SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorEnd   = pixelrechitRange.second;
      if ((pixelrechitRangeIteratorEnd-pixeliter)>0){
	
        DetSet<const SiPixelRecHit*> sipix= *(pixelInTrack.find(detid.rawId()));	
	DetSet<SiPixelCluster> Pix_second(detid.rawId()); 
	
	for ( ; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter) {   
	  int itk=0; 
	  
	  for (ipix=sipix.begin();ipix!=sipix.end();ipix++){
	    if ((*ipix)->sharesInput(&(*pixeliter) , TrackingRecHit::all )) itk++;
	  }
	  
	  if (itk==0) Pix_second.push_back((*(*pixeliter).cluster()));
	}
	if(!Pix_second.empty())  thePixelClusterVector.push_back(Pix_second);
      }
    } //end pixel detid


    //STRIP 
    if ((uint(detid.subdetId())==StripSubdetector::TIB) ||
	(uint(detid.subdetId())==StripSubdetector::TOB) ||
	(uint(detid.subdetId())==StripSubdetector::TID) ||
	(uint(detid.subdetId())==StripSubdetector::TEC)){
      
      //rphi hits
      SiStripRecHit2DCollection::range monorechitRange = (rphirecHits.product())->get((detid));
      SiStripRecHit2DCollection::const_iterator monoiter =monorechitRange.first;
      SiStripRecHit2DCollection::const_iterator monorechitRangeIteratorEnd   = monorechitRange.second;
      if ((monorechitRangeIteratorEnd-monoiter)>0){
	DetSet<SiStripCluster> Strip_second(detid.rawId()); 
	DetSet<const SiStripRecHit2D*> simono= *(monoInTrack.find(detid.rawId()));
	for ( ; monoiter != monorechitRangeIteratorEnd; ++monoiter) {
	  
	  int itk=0;
	  for (istrip=simono.begin();istrip!=simono.end();istrip++){
	    
	    if ((*istrip)->sharesInput(&(*monoiter) , TrackingRecHit::all ))itk++;
	  }
	  if (itk==0) Strip_second.push_back((*(*monoiter).cluster()));
	}
	
	if(!Strip_second.empty())  theStripClusterVector.push_back(Strip_second);
      }
      
      
      //stereo hits
      SiStripRecHit2DCollection::range stereorechitRange = (stereorecHits.product())->get((detid));
      SiStripRecHit2DCollection::const_iterator stereoiter =stereorechitRange.first;
      SiStripRecHit2DCollection::const_iterator stereorechitRangeIteratorEnd =stereorechitRange.second;
      
      if ((stereorechitRangeIteratorEnd-stereoiter)>0){
	DetSet<SiStripCluster> Strip_second(detid.rawId()); 
	DetSet<const SiStripRecHit2D*> sistereo= *(stereoInTrack.find(detid.rawId()));
	for ( ; stereoiter != stereorechitRangeIteratorEnd; ++stereoiter) {
	  int itk=0;
	  for (istrip=sistereo.begin();istrip!=sistereo.end();istrip++){
	    if ((*istrip)->sharesInput(&(*stereoiter) , TrackingRecHit::all ))itk++;
	  }
	  if (itk==0) Strip_second.push_back((*(*stereoiter).cluster()));
	} 
	if(!Strip_second.empty())  theStripClusterVector.push_back(Strip_second);
      }
    } // end strips
    
    
  } //end loop over tracker detid
  
  auto_ptr<DetSetVector<SiPixelCluster> > 
    output_pixel(new DetSetVector<SiPixelCluster>(thePixelClusterVector) );

  auto_ptr<DetSetVector<SiStripCluster> > 
    output_strip(new DetSetVector<SiStripCluster>(theStripClusterVector) );
  
  iEvent.put(output_pixel);
  iEvent.put(output_strip);
}

// ------------ method called once each job just before starting event loop  ------------
void 
RemainingClusterProducer::beginJob(const EventSetup& iSetup)
{
  iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RemainingClusterProducer::endJob() {
}


