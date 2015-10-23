#ifndef Alignment_APEEstimation_TrackerDetectorStruct_h
#define Alignment_APEEstimation_TrackerDetectorStruct_h


//#include "TH1.h"

struct TrackerDetectorStruct{
  
  TrackerDetectorStruct(): TrkSize(0), TrkSizeGood(0),
                           HitsSize(0), HitsValid(0), HitsInvalid(0), Hits2D(0),
			   HitsGood(0), LayersMissed(0),
			   HitsPixel(0), HitsStrip(0),
			   Charge(0),
			   Chi2(0), Ndof(0), NorChi2(0), Prob(0),
			   Eta(0), EtaErr(0), EtaSig(0), Theta(0),
			   Phi(0), PhiErr(0), PhiSig(0),
			   D0Beamspot(0), D0BeamspotErr(0), D0BeamspotSig(0),
			   Dz(0), DzErr(0), DzSig(0),
			   P(0), Pt(0), PtErr(0), PtSig(0),
			   MeanAngle(0),
			   HitsGoodVsHitsValid(0), MeanAngleVsHits(0),
			   HitsPixelVsEta(0), HitsPixelVsTheta(0),
			   HitsStripVsEta(0), HitsStripVsTheta(0),
			   PtVsEta(0), PtVsTheta(0),
			   PHitsGoodVsHitsValid(0), PMeanAngleVsHits(0),
			   PHitsPixelVsEta(0), PHitsPixelVsTheta(0),
			   PHitsStripVsEta(0), PHitsStripVsTheta(0),
			   PPtVsEta(0), PPtVsTheta(0){}
  
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
