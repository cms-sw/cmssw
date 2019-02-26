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

  CSCTFPtMethods( const L1MuTriggerPtScale* ptScale = nullptr ) ;

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
 
/** parameters for Anna's method 2011*/
  static const double AB_mu12FnoME11[4][15];
  static const double AB_mu13FnoME11[4][15];
  static const double AB_mu14FnoME11[4][15];
  static const double AB_mu12RnoME11[4][15];
  static const double AB_mu13RnoME11[4][15];
  static const double AB_mu14RnoME11[4][15];
  static const double AB_mu12FME11[4][15];
  static const double AB_mu13FME11[4][15];
  static const double AB_mu14FME11[4][15];
  static const double AB_mu12RME11[4][15];
  static const double AB_mu13RME11[4][15];
  static const double AB_mu14RME11[4][15];
  static const double AB_sig12FnoME11[4][15];
  static const double AB_sig13FnoME11[4][15];
  static const double AB_sig14FnoME11[4][15];
  static const double AB_sig12RnoME11[4][15];
  static const double AB_sig13RnoME11[4][15];
  static const double AB_sig14RnoME11[4][15];
  static const double AB_sig12FME11[4][15];
  static const double AB_sig13FME11[4][15];
  static const double AB_sig14FME11[4][15];
  static const double AB_sig12RME11[4][15];
  static const double AB_sig13RME11[4][15];
  static const double AB_sig14RME11[4][15];

  static const double AB_mu12F[4][15];
  static const double AB_mu13F[4][15];
  static const double AB_mu14F[4][15];
  static const double AB_mu12R[4][15];
  static const double AB_mu13R[4][15];
  static const double AB_mu14R[4][15];
  static const double AB_mu23[4][15];
  static const double AB_mu24[4][15];
  static const double AB_mu34[4][15];
  static const double AB_mu51[4][15];
  static const double AB_mu52[4][15];
  static const double AB_mu53[4][15];
  static const double AB_mu5[4][15];

  static const double AB_sig12F[4][15];
  static const double AB_sig13F[4][15];
  static const double AB_sig14F[4][15];
  static const double AB_sig12R[4][15];
  static const double AB_sig13R[4][15];
  static const double AB_sig14R[4][15];
  static const double AB_sig23[4][15];
  static const double AB_sig24[4][15];
  static const double AB_sig34[4][15];
  static const double AB_sig51[4][15];
  static const double AB_sig52[4][15];
  static const double AB_sig53[4][15];
  static const double AB_sig5[4][15];

  static const double AB_rho123F[5][15];
  static const double AB_rho124F[5][15];
  static const double AB_rho134F[5][15];
  static const double AB_rho123R[5][15];
  static const double AB_rho124R[5][15];
  static const double AB_rho134R[5][15];
  static const double AB_rho234[5][15];
  static const double AB_rho512[5][15];
  static const double AB_rho513[5][15];
  static const double AB_rho523[5][15];
  static const double AB_rho51B[5][15];
  static const double AB_rho52B[5][15];
  static const double AB_rho53B[5][15];
/** parameters for Anna's method 2010*/
  static const double A_mu12Front[4][15];
  static const double A_sig12Front[3][15];
  static const double A_mu13Front[4][15];
  static const double A_sig13Front[3][15];
  static const double A_mu14Front[4][15];
  static const double A_sig14Front[3][15];

  static const double A_mu12Rare[4][15];
  static const double A_sig12Rare[3][15];
  static const double A_mu13Rare[4][15];
  static const double A_sig13Rare[3][15];
  static const double A_mu14Rare[4][15];
  static const double A_sig14Rare[3][15];

  static const double A_mu51[4][15];
  static const double A_sig51[3][15];
  static const double A_mu52[4][15];
  static const double A_sig52[3][15];
  static const double A_mu53[4][15];
  static const double A_sig53[3][15];

  static const double A_mu23[4][15];
  static const double A_sig23[3][15];
  static const double A_mu24[4][15];
  static const double A_sig24[3][15];
  static const double A_mu34[4][15];
  static const double A_sig34[3][15];

/*
  static const double A_mu23CSCTF[4][15];
  static const double A_sig23CSCTF[3][15];
  static const double A_mu24CSCTF[4][15];
  static const double A_sig24CSCTF[3][15];
  static const double A_mu34CSCTF[4][15];
  static const double A_sig34CSCTF[3][15];
*/ 
  static const double A_rho123FrontCSCTF[5][15];
  static const double A_rho124FrontCSCTF[5][15];
  static const double A_rho134FrontCSCTF[5][15];

  static const double A_rho123RareCSCTF[5][15];
  static const double A_rho124RareCSCTF[5][15];
  static const double A_rho134RareCSCTF[5][15];

  static const double A_rho234CSCTF[5][15];

  static const double A_rho512[5][15];
  static const double A_rho513[5][15];
  static const double A_rho523[5][15];

// don't care about Mode 12: 1-2-b1 yet, should add A_mu12CSCTF or A_mu51CSCTF depending how calculate dphi12

  /** 2-station Pt measurement for types (see SP class for 2-stn types) */
  float Pt2Stn(int type, float eta, float dphi, int fr=-1) const;
  float Pt2Stn2010(int type, float eta, float dphi, int fr=-1, int method=11) const;
  float Pt2Stn2011(int type, float eta, float dphi, int fr=-1, int method=11, int phiSign=2) const;
  float Pt2Stn2012(int type, float eta, float dphi, int PtbyMLH, float &theLH, int fr=-1, int method=11) const;
  float Pt2Stn2012_DT(int type, float eta, float dphi, int PtbyMLH, float &theLH,int fr=-1, int method=11, int phiSign=2) const;
  double Likelihood2(double *phi12, double *par_m12, double *par_sig12, double *v) const;  
  double Likelihood2_2011(double *phi12, double *par_m12, double *par_sig12, double *v) const;
 
  /** 3-station Pt measurement for types (see SP class for 3-stn types) */
  float Pt3Stn(int type, float eta, float dphi1, float dphi2, int fr=-1) const;
  float Pt3Stn2010(int type, float eta, float dphi1, float dphi2, int fr=-1, int method=11) const;
  float Pt3Stn2011(int type, float eta, float dphi1, float dphi2, int fr=-1, int method=11) const;
  float Pt3Stn2012(int type, float eta, float dphi1, float dphi2, int PtbyMLH, float &theLH, int fr=-1, int method=11) const;
  float Pt3Stn2012_DT(int type, float eta, float dphi1, float dphi2, int PtbyMLH, float &theLH, int fr=-1, int method=11) const;
  double Likelihood(double *phi12, double *phi23, double *par_m12, double *par_m23, double *par_sig12, double *par_sig23, double *par_rho, double *v) const; 
  double Likelihood2011(double *phi12, double *phi23, double *par_m12, double *par_m23, double *par_sig12, double *par_sig23, double *par_rho, double *v) const;
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
