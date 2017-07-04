#ifndef Alignment_APEEstimation_TrackerDetectorStruct_h
#define Alignment_APEEstimation_TrackerDetectorStruct_h


//#include "TH1.h"

struct TrackerDetectorStruct{
  
  TrackerDetectorStruct(): TrkSize(nullptr), TrkSizeGood(nullptr),
                           HitsSize(nullptr), HitsValid(nullptr), HitsInvalid(nullptr), Hits2D(nullptr),
			   HitsGood(nullptr), LayersMissed(nullptr),
			   HitsPixel(nullptr), HitsStrip(nullptr),
			   Charge(nullptr),
			   Chi2(nullptr), Ndof(nullptr), NorChi2(nullptr), Prob(nullptr),
			   Eta(nullptr), EtaErr(nullptr), EtaSig(nullptr), Theta(nullptr),
			   Phi(nullptr), PhiErr(nullptr), PhiSig(nullptr),
			   D0Beamspot(nullptr), D0BeamspotErr(nullptr), D0BeamspotSig(nullptr),
			   Dz(nullptr), DzErr(nullptr), DzSig(nullptr),
			   P(nullptr), Pt(nullptr), PtErr(nullptr), PtSig(nullptr),
			   MeanAngle(nullptr),
			   HitsGoodVsHitsValid(nullptr), MeanAngleVsHits(nullptr),
			   HitsPixelVsEta(nullptr), HitsPixelVsTheta(nullptr),
			   HitsStripVsEta(nullptr), HitsStripVsTheta(nullptr),
			   PtVsEta(nullptr), PtVsTheta(nullptr),
			   PHitsGoodVsHitsValid(nullptr), PMeanAngleVsHits(nullptr),
			   PHitsPixelVsEta(nullptr), PHitsPixelVsTheta(nullptr),
			   PHitsStripVsEta(nullptr), PHitsStripVsTheta(nullptr),
			   PPtVsEta(nullptr), PPtVsTheta(nullptr){}
  
  TH1 *TrkSize, *TrkSizeGood,
      *HitsSize, *HitsValid, *HitsInvalid, *Hits2D,
      *HitsGood, *LayersMissed,
      *HitsPixel, *HitsStrip,
      *Charge,
      *Chi2, *Ndof, *NorChi2, *Prob,
      *Eta, *EtaErr, *EtaSig, *Theta,
      *Phi,*PhiErr, *PhiSig,
      *D0Beamspot, *D0BeamspotErr, *D0BeamspotSig,
      *Dz, *DzErr, *DzSig,
      *P, *Pt, *PtErr, *PtSig,
      *MeanAngle;
  
  TH2 *HitsGoodVsHitsValid, *MeanAngleVsHits,
      *HitsPixelVsEta, *HitsPixelVsTheta,
      *HitsStripVsEta, *HitsStripVsTheta,
      *PtVsEta, *PtVsTheta;
  
  TProfile *PHitsGoodVsHitsValid, *PMeanAngleVsHits,
           *PHitsPixelVsEta, *PHitsPixelVsTheta,
           *PHitsStripVsEta, *PHitsStripVsTheta,
           *PPtVsEta, *PPtVsTheta;
  
};

#endif
