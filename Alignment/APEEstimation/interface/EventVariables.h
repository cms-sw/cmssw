#ifndef Alignment_APEEstimation_EventVariables_h
#define Alignment_APEEstimation_EventVariables_h


struct TrackStruct{
  
  TrackStruct(){}
  
  enum HitState{notInTracker, notAssignedToSectors, invalid, negativeError, ok};
  
  struct HitParameterStruct{
    
    HitParameterStruct(): hitState(ok),
                 isPixelHit(false),
		 goodXMeasurement(false),
		 goodYMeasurement(false),
		 widthX(0),
		 baryStripX(-999.F),
		 widthY(0),
		 baryStripY(-999.F),
		 chargePixel(-999.F),
		 clusterProbabilityXY(-999.F), clusterProbabilityQ(-999.F),
		 clusterProbabilityXYQ(-999.F), logClusterProbability(-999.F),
		 isOnEdge(false), hasBadPixels(false), spansTwoRoc(false),
		 qBin(-1),
		 isModuleUsable(true),
		 chargeStrip(0),
		 maxStrip(0), maxStripInv(0), maxCharge(0), maxIndex(0),
		 chargeOnEdges(-999.F), chargeAsymmetry(-999.F), 
		 chargeLRplus(-999.F), chargeLRminus(-999.F),
		 sOverN(-999.F),
		 projWidth(-999.F),
		 resX(-999.F), norResX(-999.F), xHit(-999.F), xTrk(-999.F),
                 errXHit(-999.F), errXTrk(-999.F), errX(-999.F), errX2(-999.F),
		 errXHitWoApe(-999.F), errXWoApe(-999.F),
		 probX(-999.F),
		 resY(-999.F), norResY(-999.F), yHit(-999.F), yTrk(-999.F),
                 errYHit(-999.F), errYTrk(-999.F), errY(-999.F), errY2(-999.F),
		 errYHitWoApe(-999.F), errYWoApe(-999.F),
		 probY(-999.F),
		 phiSens(-999.F), phiSensX(-999.F), phiSensY(-999.F){}
    
    HitState hitState;
    bool isPixelHit;
    bool goodXMeasurement, goodYMeasurement;
    std::vector<unsigned int> v_sector;
    
    // Cluster parameters
    // pixel+strip
    unsigned int widthX;
    float baryStripX;
    // pixel only
    unsigned int widthY;
    float baryStripY;
    float chargePixel;
    float clusterProbabilityXY, clusterProbabilityQ,
          clusterProbabilityXYQ, logClusterProbability;
    bool isOnEdge, hasBadPixels, spansTwoRoc;
    int qBin;
    // strip only
    bool isModuleUsable;
    unsigned int chargeStrip;
    unsigned int maxStrip, maxStripInv, maxCharge, maxIndex;
    float chargeOnEdges, chargeAsymmetry,
          chargeLRplus, chargeLRminus;
    float sOverN;
    float projWidth;
    
    // trackFit results
    float resX, norResX, xHit, xTrk,
          errXHit, errXTrk, errX, errX2,
	  errXHitWoApe, errXWoApe,
	  probX;
    float resY, norResY, yHit, yTrk,
          errYHit, errYTrk, errY, errY2,
	  errYHitWoApe, errYWoApe,
	  probY;
    float phiSens, phiSensX, phiSensY;
  };
  
  struct TrackParameterStruct{
    
    TrackParameterStruct(): hitsSize(-999), hitsValid(-999), hitsInvalid(-999),
		   hits2D(-999), layersMissed(-999),
		   hitsPixel(-999), hitsStrip(-999),
		   charge(-999),
		   chi2(-999.F), ndof(-999.F), norChi2(-999.F), prob(-999.F),
                   eta(-999.F), etaErr(-999.F), theta(-999.F),
		   phi(-999.F), phiErr(-999.F),
		   d0(-999.F), d0Beamspot(-999.F), d0BeamspotErr(-999.F),
		   dz(-999.F), dzErr(-999.F), dzBeamspot(-999.F),
                   p(-999.F), pt(-999.F), ptErr(-999.F),
		   meanPhiSensToNorm(-999.F){}
    
    int hitsSize, hitsValid, hitsInvalid,
        hits2D, layersMissed,
	hitsPixel, hitsStrip,
	charge;
    float chi2, ndof, norChi2, prob,
          eta, etaErr, theta,
	  phi, phiErr,
	  d0, d0Beamspot, d0BeamspotErr,
	  dz, dzErr, dzBeamspot,
	  p, pt, ptErr,
	  meanPhiSensToNorm;
  };
  
  TrackParameterStruct trkParams;
  std::vector<HitParameterStruct> v_hitParams;
};

#endif
