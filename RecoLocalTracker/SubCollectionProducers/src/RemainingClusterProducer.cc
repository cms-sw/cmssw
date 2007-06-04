#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
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
  double cut = conf_.getParameter<double>("DistanceCut");

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

  //TRACKING HITS
  vector<const SiPixelRecHit*> pixelInTrack;
  vector<const SiStripMatchedRecHit2D*> matchedInTrack;
  vector<const SiStripRecHit2D*> monoInTrack;
  //

  TrackingRecHitCollection::const_iterator hit;
  for(hit=trackingHits.product()->begin();hit!=trackingHits.product()->end();hit++){
    const SiPixelRecHit* pixelhit=dynamic_cast<const SiPixelRecHit*>(&(*hit));
    if (pixelhit!=0)  pixelInTrack.push_back(pixelhit);
    const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*hit));
    if (matchedhit!=0)  matchedInTrack.push_back(matchedhit);
    const SiStripRecHit2D* monohit=dynamic_cast<const SiStripRecHit2D*>(&(*hit)); 
    if (monohit!=0) monoInTrack.push_back(monohit);
  }
  vector<const SiPixelRecHit*>::const_iterator ipix;
  vector<const SiStripRecHit2D*>::const_iterator imono;
  vector<const SiStripMatchedRecHit2D*>::const_iterator imatched;

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
      DetSet<SiPixelCluster> Pix_second(detid.rawId()); 
      for ( ; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter) {   
	int itk=0;
	for (ipix=pixelInTrack.begin();ipix!=pixelInTrack.end();ipix++){
	  
	  if (((*ipix)->geographicalId()==(*pixeliter).geographicalId())&&
	      (((*ipix)->localPosition()-(*pixeliter).localPosition()).mag()<cut)) itk++;
	}
	
	if (itk==0) Pix_second.push_back((*(*pixeliter).cluster()));
      }
      if(!Pix_second.empty())  thePixelClusterVector.push_back(Pix_second);
    } //end pixel detid

    //STRIP SINGLE SIDED
    if (((uint(detid.subdetId())==StripSubdetector::TIB) && (TIBDetId(detid).layer()>2)) ||
	((uint(detid.subdetId())==StripSubdetector::TOB) && (TOBDetId(detid).layer()>2)) ||
	((uint(detid.subdetId())==StripSubdetector::TID) && (TIDDetId(detid).ring()>2))  ||
	((uint(detid.subdetId())==StripSubdetector::TEC) && (TECDetId(detid).ring()>2) &&
	 (TECDetId(detid).ring()!=5))){
    
      DetSet<SiStripCluster> Strip_second(detid.rawId()); 
      SiStripRecHit2DCollection::range monorechitRange = (rphirecHits.product())->get((detid));
      SiStripRecHit2DCollection::const_iterator monoiter =monorechitRange.first;
      SiStripRecHit2DCollection::const_iterator monorechitRangeIteratorEnd   = monorechitRange.second;
   
      for ( ; monoiter != monorechitRangeIteratorEnd; ++monoiter) {

	int itk=0;
	for (imono=monoInTrack.begin();imono!=monoInTrack.end();imono++){

	  if (((*imono)->geographicalId()==(*monoiter).geographicalId())&&
	      (((*imono)->localPosition()-(*monoiter).localPosition()).mag()<cut))itk++;

	}

	if (itk==0) Strip_second.push_back((*(*monoiter).cluster()));
      }
   
      if(!Strip_second.empty())  theStripClusterVector.push_back(Strip_second);
    } // end single sided strip detid

    //STRIP DOUBLE SIDED
   if ((uint(detid.subdetId())==StripSubdetector::TOB)&&(TOBDetId(detid).layer()<3) ||
	(uint(detid.subdetId())==StripSubdetector::TIB)&&(TIBDetId(detid).layer()<3) ||
	(uint(detid.subdetId())==StripSubdetector::TEC)&&(TECDetId(detid).ring()<3) ||
	(uint(detid.subdetId())==StripSubdetector::TEC)&&(TECDetId(detid).ring()==5) ||
	(uint(detid.subdetId())==StripSubdetector::TID)&&(TIDDetId(detid).ring()<3)) {
     DetSet<SiStripCluster> Strip_second(detid.rawId()); 
   
     //Rphi hits
     SiStripRecHit2DCollection::range monorechitRange = (rphirecHits.product())->get((detid));
     SiStripRecHit2DCollection::const_iterator monoiter =monorechitRange.first;
     SiStripRecHit2DCollection::const_iterator monorechitRangeIteratorEnd =monorechitRange.second;
     for ( ; monoiter != monorechitRangeIteratorEnd; ++monoiter) {
           int itk=0;
       for (imatched=matchedInTrack.begin();imatched!=matchedInTrack.end();imatched++){
	 
	 if(((((*imatched))->monoHit())->geographicalId()==(*monoiter).geographicalId()) &&
	    ((((*imatched)->monoHit())->localPosition()-
	      (*monoiter).localPosition()).mag()<cut)) itk++;
       }
       if (itk==0) Strip_second.push_back((*(*monoiter).cluster()));
     }     
   
     //stereo hits
     SiStripRecHit2DCollection::range stereorechitRange = (stereorecHits.product())->get((detid));
     SiStripRecHit2DCollection::const_iterator stereoiter =stereorechitRange.first;
     SiStripRecHit2DCollection::const_iterator stereorechitRangeIteratorEnd =stereorechitRange.second;
     for ( ; stereoiter != stereorechitRangeIteratorEnd; ++stereoiter) {
           int itk=0;
       for (imatched=matchedInTrack.begin();imatched!=matchedInTrack.end();imatched++){
	 
	 if(((((*imatched))->stereoHit())->geographicalId()==(*stereoiter).geographicalId()) &&
	    ((((*imatched)->stereoHit())->localPosition()-
	      (*stereoiter).localPosition()).mag()<cut)) itk++;
       }
       if (itk==0) Strip_second.push_back((*(*stereoiter).cluster()));
     } 
     if(!Strip_second.empty())  theStripClusterVector.push_back(Strip_second);
     
   } // end double sided strip detid

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


