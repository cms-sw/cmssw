#ifndef CSCTrackFinder_CSCTFPtMethods_h
#define CSCTrackFinder_CSCTFPtMethods_h

#include <CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h>

class L1MuTriggerPtScale ;

class CSCTFPtMethods
{
 public:
  enum {kMaxParameters = 4};  
  /** Allowed station combinations for extrapolation units */
  enum {kME1andME2=1, kME1andME3, kME2andME3, kME2andME4, 
	kME3andME4, kME1andME2ovr, kME2andMB1, kME2andMB2};

  CSCTFPtMethods( const L1MuTriggerPtScale* ptScale = 0 ) ;

  /** First is the parameterizations of Acosta/McDonald */
  
  /** The two station pt measument needs a two constant fit with a break in the detector depending on what detectors it hit (eta) */
  /** dphi = A/pt + B/(pt^2) */ 
  static const float AkLowEta_Fit2[kME2andMB2][kMaxParameters];
  static const float AkHighEta_Fit2[kME2andMB2][kMaxParameters];
  
  static const float BkLowEta_Fit2[kME2andMB2][kMaxParameters];
  static const float BkHighEta_Fit2[kME2andMB2][kMaxParameters];
  
  /** The three station pt measument only needs a one constant fit, but the dependence on eta is still there */
  /** dphi = A/pt */
  static const float AkLowEta_Fit1[kME2andMB2][kMaxParameters];
  static const float AkHighEta_Fit1[kME2andMB2][kMaxParameters];
  
  static const float kGlobalScaleFactor;
  /** Corrections for ME1 F/R bit */
  static const float FRCorrLowEta[kME2andMB2][2];
  static const float FRCorrHighEta[kME2andMB2][2];
  
  /** 2-station Pt measurement for types (see SP class for 2-stn types) */
  float Pt2Stn(int type, float eta, float dphi, int fr=-1) const;
  
  /** 3-station Pt measurement for types (see SP class for 3-stn types) */
  float Pt3Stn(int type, float eta, float dphi1, float dphi2, int fr=-1) const;
    
  /** Second are the parameterizations of Acosta/Yeh */

  static const float ptbins[29];
  static const float etabins[16];
  static const float dphifr0[4][15][28];
  static const float dphifr1[4][15][28];
  static const float sigmafr0[4][15][28];
  static const float sigmafr1[4][15][28];

  float Pt2StnChiSq(int type, float eta, int dphi, int fr) const;
  float Pt3StnChiSq(int type, float eta, int dphi1, int dphi2, int fr) const;

  /** Third is the hybrid method */

  float Pt2StnHybrid(int type, float eta, int dphi, int fr) const;
  float Pt3StnHybrid(int type, float eta, int dphi1, int dphi2, int fr) const;

  /** The hybrid method may be changing soon to:
   *  1st Calculate PT with Darin's method
   *  2nd if BELOW a certain cut call Cathy's method
   *  3rd if Cathy's < Darin's use Cathy's otherwise return Darin's
   *  A study needs to be performed to determine any gains from this procedure.
   */
  
  /** A method to calculate the charge valid bit
   *  Regions where this bit is were determined via simulation
   */
  bool chargeValid(unsigned Pt, unsigned Quality, unsigned Eta, unsigned method) const;

  /** Legacy Pt90 calculation function */
  float PtEff90(float pt, float eta, int mode) const;

 private:  
  const L1MuTriggerPtScale* trigger_scale;
};

#endif
