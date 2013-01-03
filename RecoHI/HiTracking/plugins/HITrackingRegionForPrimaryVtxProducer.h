#ifndef RecoHI_HiTracking_HITrackingRegionForPrimaryVtxProducer_H 
#define RecoHI_HiTracking_HITrackingRegionForPrimaryVtxProducer _H

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h" 	 
#include "DataFormats/Common/interface/DetSetAlgorithm.h"

#include "DataFormats/Common/interface/DetSetVector.h"    
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "TMath.h"

class HITrackingRegionForPrimaryVtxProducer : public TrackingRegionProducer {
  
 public:
  
  HITrackingRegionForPrimaryVtxProducer(const edm::ParameterSet& cfg) { 
    
    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");
    thePtMin            = regionPSet.getParameter<double>("ptMin");
    theOriginRadius     = regionPSet.getParameter<double>("originRadius");
    theNSigmaZ          = regionPSet.getParameter<double>("nSigmaZ");
    theBeamSpotTag      = regionPSet.getParameter<edm::InputTag>("beamSpot");
    thePrecise          = regionPSet.getParameter<bool>("precise"); 
    theSiPixelRecHits   = regionPSet.getParameter<edm::InputTag>("siPixelRecHits");  
    doVariablePtMin     = regionPSet.getParameter<bool>("doVariablePtMin"); 
    double xDir         = regionPSet.getParameter<double>("directionXCoord");
    double yDir         = regionPSet.getParameter<double>("directionYCoord");
    double zDir         = regionPSet.getParameter<double>("directionZCoord");
    theDirection = GlobalVector(xDir, yDir, zDir);

    // for using vertex instead of beamspot
    theSigmaZVertex     = regionPSet.getParameter<double>("sigmaZVertex");
    theFixedError       = regionPSet.getParameter<double>("fixedError");
    theUseFoundVertices = regionPSet.getParameter<bool>("useFoundVertices");
    theUseFixedError    = regionPSet.getParameter<bool>("useFixedError");
    vertexCollName      = regionPSet.getParameter<edm::InputTag>("VertexCollection");
  }   
  
  virtual ~HITrackingRegionForPrimaryVtxProducer(){}
  
  int estimateMultiplicity
    (const edm::Event& ev, const edm::EventSetup& es) const
    {
      //rechits
      edm::Handle<SiPixelRecHitCollection> recHitColl;
      ev.getByLabel(theSiPixelRecHits, recHitColl);
      
      std::vector<const TrackingRecHit*> theChosenHits; 	 
      TrackerLayerIdAccessor acc; 	 
      edmNew::copyDetSetRange(*recHitColl,theChosenHits,acc.pixelBarrelLayer(1)); 	 
      return theChosenHits.size(); 	 
      
    }
  
  virtual std::vector<TrackingRegion* > regions(const edm::Event& ev, const edm::EventSetup& es) const {
    
    int estMult = estimateMultiplicity(ev, es);
    
    // from MC relating first layer pixel hits to "findable" sim tracks with pt>1 GeV
    float cc = -38.6447;
    float bb = 0.0581765;
    float aa = 1.34306e-06;

    float estTracks = aa*estMult*estMult+bb*estMult+cc;
    
    LogTrace("heavyIonHLTVertexing")<<"[HIVertexing]";
    LogTrace("heavyIonHLTVertexing")<<" [HIVertexing: hits in the 1. layer:" << estMult << "]";
    LogTrace("heavyIonHLTVertexing")<<" [HIVertexing: estimated number of tracks:" << estTracks << "]";
    
    float regTracking = 60.;  //if we have more tracks -> regional tracking
    float etaB = 10.;
    float phiB = TMath::Pi()/2.;
    
    float decEta = estTracks/90.;
    etaB = 2.5/decEta;
    
    if(estTracks>regTracking) {
      LogTrace("heavyIonHLTVertexing")<<" [HIVertexing: Regional Tracking]";
      LogTrace("heavyIonHLTVertexing")<<"  [Regional Tracking: eta range: -" << etaB << ", "<< etaB <<"]";
      LogTrace("heavyIonHLTVertexing")<<"  [Regional Tracking: phi range: -" << phiB << ", "<< phiB <<"]";
      LogTrace("heavyIonHLTVertexing")<<"  [Regional Tracking: factor of decrease: " << decEta*2. << "]";  // 2:from phi
    }
    
    float minpt = thePtMin;
    float varPtCutoff = 1500; //cutoff
    if(doVariablePtMin && estMult < varPtCutoff) {
      minpt = 0.075;
      if(estMult > 0) minpt += estMult * (thePtMin - 0.075)/varPtCutoff; // lower ptMin linearly with pixel hit multiplicity
    }
    
    // tracking region selection
    std::vector<TrackingRegion* > result;
    double halflength;
    GlobalPoint origin;
    edm::Handle<reco::BeamSpot> bsHandle;
    ev.getByLabel( theBeamSpotTag, bsHandle);
    if(bsHandle.isValid()) {
      const reco::BeamSpot & bs = *bsHandle; 
      origin=GlobalPoint(bs.x0(), bs.y0(), bs.z0()); 
      halflength=theNSigmaZ*bs.sigmaZ();
      
    if(theUseFoundVertices)
    {
      edm::Handle<reco::VertexCollection> vertexCollection;
      ev.getByLabel(vertexCollName,vertexCollection);

      for(reco::VertexCollection::const_iterator iV=vertexCollection->begin(); 
	  iV != vertexCollection->end() ; iV++) {
          if (iV->isFake() || !iV->isValid()) continue;
	  origin     = GlobalPoint(bs.x0(),bs.y0(),iV->z());
	  halflength = (theUseFixedError ? theFixedError : (iV->zError())*theSigmaZVertex); 
      }
    }

      if(estTracks>regTracking) {  // regional tracking
        result.push_back( 
	  new RectangularEtaPhiTrackingRegion(theDirection, origin, thePtMin, theOriginRadius, halflength, etaB, phiB, 0, thePrecise) );
      }
      else {                       // global tracking
        LogTrace("heavyIonHLTVertexing")<<" [HIVertexing: Global Tracking]";
        result.push_back( 
	  new GlobalTrackingRegion(minpt, origin, theOriginRadius, halflength, thePrecise) );
      }
    } 
    return result;
  }
  
 private:
  double thePtMin; 
  double theOriginRadius; 
  double theNSigmaZ;
  edm::InputTag theBeamSpotTag;	
  bool thePrecise;
  GlobalVector theDirection;
  edm::InputTag theSiPixelRecHits;
  bool doVariablePtMin;

  double theSigmaZVertex;
  double theFixedError;  
  bool theUseFoundVertices;
  bool theUseFixedError;
  edm::InputTag vertexCollName;


};

#endif 
