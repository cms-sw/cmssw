#ifndef RecoHI_HiTracking_HITrackingRegionProducer_H 
#define RecoHI_HiTracking_HITrackingRegionProducer_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"    
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TMath.h"

class HITrackingRegionProducer : public TrackingRegionProducer {
  
 public:
  
  HITrackingRegionProducer(const edm::ParameterSet& cfg, edm::ConsumesCollector && iC) { 
    
    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");
    
    thePtMin            = regionPSet.getParameter<double>("ptMin");
    theOriginRadius     = regionPSet.getParameter<double>("originRadius");
    theOriginHalfLength = regionPSet.getParameter<double>("originHalfLength");
    double xPos         = regionPSet.getParameter<double>("originXPos");
    double yPos         = regionPSet.getParameter<double>("originYPos");
    double zPos         = regionPSet.getParameter<double>("originZPos");
    double xDir         = regionPSet.getParameter<double>("directionXCoord");
    double yDir         = regionPSet.getParameter<double>("directionYCoord");
    double zDir         = regionPSet.getParameter<double>("directionZCoord");
    thePrecise          = regionPSet.getParameter<bool>("precise"); 
    theSiPixelRecHits   = regionPSet.getParameter<edm::InputTag>("siPixelRecHits");
    theSiPixelRecHitsToken = iC.consumes<SiPixelRecHitCollection> (theSiPixelRecHits);
    theOrigin = GlobalPoint(xPos,yPos,zPos);
    theDirection = GlobalVector(xDir, yDir, zDir);
  }   
  
  virtual ~HITrackingRegionProducer(){}
  
  int estimateMultiplicity
    (const edm::Event& ev, const edm::EventSetup& es) const
  {
    //rechits
    edm::Handle<SiPixelRecHitCollection> recHitColl;
    ev.getByToken(theSiPixelRecHitsToken, recHitColl);
    
    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopo;
    es.get<IdealGeometryRecord>().get(tTopo);
    
    int numRecHits = 0;
    //FIXME: this can be optimized quite a bit by looping only on the per-det 'items' of DetSetVector
    for(SiPixelRecHitCollection::const_iterator recHitIdIterator = recHitColl->begin(), recHitIdIteratorEnd = recHitColl->end();
	recHitIdIterator != recHitIdIteratorEnd; recHitIdIterator++) {
      SiPixelRecHitCollection::DetSet hits = *recHitIdIterator;
      DetId detId = DetId(hits.detId());   // Get the Detid object
      unsigned int detType=detId.det();    // det type, tracker=1
      unsigned int subid=detId.subdetId(); //subdetector type, barrel=1, fpix=2
      
      unsigned int layer=0;
      layer=tTopo->pxbLayer(detId);
      if(detType==1 && subid==1 && layer==1) {
	numRecHits += hits.size();
      }
    }
    return numRecHits;
  }
  
  virtual std::vector<TrackingRegion* > regions(const edm::Event& ev, const edm::EventSetup& es) const {
    
    int estMult = estimateMultiplicity(ev, es);
    
    // fit from MC information
    float aa = 1.90935e-04;
    float bb = -2.90167e-01;
    float cc = 3.86125e+02;
    
    float estTracks = aa*estMult*estMult+bb*estMult+cc;
    
    LogTrace("heavyIonHLTVertexing")<<"[HIVertexing]";
    LogTrace("heavyIonHLTVertexing")<<" [HIVertexing: hits in the 1. layer:" << estMult << "]";
    LogTrace("heavyIonHLTVertexing")<<" [HIVertexing: estimated number of tracks:" << estTracks << "]";
    
    float regTracking = 400.;  //if we have more tracks -> regional tracking
    float etaB = 10.;
    float phiB = TMath::Pi()/2.;
    
    float decEta = estTracks/600.;
    etaB = 2.5/decEta;
    
    if(estTracks>regTracking) {
      LogTrace("heavyIonHLTVertexing")<<" [HIVertexing: Regional Tracking]";
      LogTrace("heavyIonHLTVertexing")<<"  [Regional Tracking: eta range: -" << etaB << ", "<< etaB <<"]";
      LogTrace("heavyIonHLTVertexing")<<"  [Regional Tracking: phi range: -" << phiB << ", "<< phiB <<"]";
      LogTrace("heavyIonHLTVertexing")<<"  [Regional Tracking: factor of decrease: " << decEta*2. << "]";  // 2:from phi
    }
    
    // tracking region selection
    std::vector<TrackingRegion* > result;
    if(estTracks>regTracking) {  // regional tracking
      result.push_back( new RectangularEtaPhiTrackingRegion(theDirection, theOrigin, thePtMin, theOriginRadius, theOriginHalfLength, etaB, phiB, RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever, thePrecise) );
    }
    else {                       // global tracking
      LogTrace("heavyIonHLTVertexing")<<" [HIVertexing: Global Tracking]";
      result.push_back( new GlobalTrackingRegion(thePtMin, theOrigin, theOriginRadius, theOriginHalfLength, thePrecise) );
    }
    return 
      result;
  }
  
 private:
  edm::InputTag theSiPixelRecHits;
  edm::EDGetTokenT<SiPixelRecHitCollection> theSiPixelRecHitsToken;
  double thePtMin; 
  GlobalPoint theOrigin;
  double theOriginRadius; 
  double theOriginHalfLength; 
  bool thePrecise;
  GlobalVector theDirection;
};

#endif 
