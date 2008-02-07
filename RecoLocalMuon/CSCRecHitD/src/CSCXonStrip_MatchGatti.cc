// This is CSCXonStrip_MatchGatti.cc

//---- Large part is copied from RecHitB
//---- author: Stoyan Stoynev - NU

#include <RecoLocalMuon/CSCRecHitD/interface/CSCXonStrip_MatchGatti.h>
#include <RecoLocalMuon/CSCRecHitD/interface/CSCStripCrosstalk.h>
#include <RecoLocalMuon/CSCRecHitD/interface/CSCStripNoiseMatrix.h>
#include <RecoLocalMuon/CSCRecHitD/interface/CSCFindPeakTime.h>
#include <RecoLocalMuon/CSCRecHitD/interface/CSCStripHit.h>

#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>

#include <CondFormats/CSCObjects/interface/CSCDBGains.h>
#include <CondFormats/DataRecord/interface/CSCDBGainsRcd.h>
#include <CondFormats/CSCObjects/interface/CSCDBCrosstalk.h>
#include <CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h>
#include <CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h> 
#include <CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h>


#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>


#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream> 
#include <iomanip.h> 
                                                                                                 
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
                                                                                                 

CSCXonStrip_MatchGatti::CSCXonStrip_MatchGatti(const edm::ParameterSet& ps){

  debug                      = ps.getUntrackedParameter<bool>("CSCDebug");
  useCalib                   = ps.getUntrackedParameter<bool>("CSCUseCalibrations");
  adcSystematics             = ps.getUntrackedParameter<double>("CSCCalibrationSystematics");        
  xtalksOffset               = ps.getUntrackedParameter<double>("CSCStripxtalksOffset");
  xtalksSystematics          = ps.getUntrackedParameter<double>("CSCStripxtalksSystematics");
  stripCrosstalk_            = new CSCStripCrosstalk( ps );
  stripNoiseMatrix_          = new CSCStripNoiseMatrix( ps );
  NoiseLevel                 = ps.getUntrackedParameter<double>("NoiseLevel"); 
  XTasymmetry                = ps.getUntrackedParameter<double>("XTasymmetry"); 
  ConstSyst                   = ps.getUntrackedParameter<double>("ConstSyst"); 
  peakTimeFinder_            = new CSCFindPeakTime();
  getCorrectionValues("StringCurrentlyNotUsed");
}


CSCXonStrip_MatchGatti::~CSCXonStrip_MatchGatti(){
  delete stripCrosstalk_;
  delete stripNoiseMatrix_;
  delete peakTimeFinder_;
}


/* findPosition
 *
 */
void CSCXonStrip_MatchGatti::findXOnStrip( const CSCDetId& id, const CSCLayer* layer, const CSCStripHit& stripHit, 
                                            int centralStrip, float& xCenterStrip, float& sWidth, 
                                            double& xGatti, float& tpeak, double& sigma, float& chisq, float& Charge ) {
  // Initialize Gatti parameters using chamberSpecs
  // Cache specs_ info for ease of access
  specs_ = layer->chamber()->specs();
  stripWidth = sWidth;
  //initChamberSpecs();

  // Initialize output parameters  
  xGatti = xCenterStrip;  
  sigma = chisq = 9999.;

  CSCStripHit::ChannelContainer strips = stripHit.strips();
  int nStrips = strips.size();
  int CenterStrip = nStrips/2 + 1;   
  std::vector<float> adcs = stripHit.s_adc();
  int tmax = stripHit.tmax();

  // Fit peaking time only if using calibrations
  float t_peak = tpeak;
  float t_zero = 0.;
  float adc[4];

  if ( useCalib ) {
    bool useFittedCharge = false;
    for ( int t = 0; t < 4; ++t ) {
      int k  = t + 4 * (CenterStrip-1);
      adc[t] = adcs[k];
    }
    useFittedCharge = peakTimeFinder_->FindPeakTime( tmax, adc, t_zero, t_peak );  // Clumsy...  can remove t_peak ?
    tpeak = t_peak+t_zero;
  }
      
  //---- fill the charge matrix (3x3)
  int j = 0;
  for ( int i = 1; i <= nStrips; ++i ) {
    if ( i > (CenterStrip-2) && i < (CenterStrip+2) ) {
      std::vector<float> adcsFit;
      for ( int t = 0; t < 4; ++t ) {
        int k  = t + 4*(i-1);
        adc[t] = adcs[k];
        if ( t < 3) ChargeSignal[j][t] = adc[t];
      }
      j++;
    }
  }


  // Load in x-talks:

  if ( useCalib ) {
    std::vector<float> xtalks;
    stripCrosstalk_->setCrossTalk( xtalk_ );
    stripCrosstalk_->getCrossTalk( id, centralStrip, xtalks);
    float dt = 50. * tmax - (t_peak + t_zero);  // QUESTION:  should it be only - (t_peak) ???
    // XTalks; l,r are for left, right XTalk; lr0,1,2 are for what charge "remains" in the strip 
    float Coef = 0.50;
    for ( int t = 0; t < 3; ++t ) {
      xt_l[0][t] = Coef*(xtalks[0] * (50.* (t-1) + dt) + xtalks[1] + xtalksOffset);
      xt_r[0][t] = Coef*(xtalks[2] * (50.* (t-1) + dt) + xtalks[3] + xtalksOffset);
      xt_l[1][t] = Coef*(xtalks[4] * (50.* (t-1) + dt) + xtalks[5] + xtalksOffset);
      xt_r[1][t] = Coef*(xtalks[6] * (50.* (t-1) + dt) + xtalks[7] + xtalksOffset);
      xt_l[2][t] = Coef*(xtalks[8] * (50.* (t-1) + dt) + xtalks[9] + xtalksOffset);
      xt_r[2][t] = Coef*(xtalks[10]* (50.* (t-1) + dt) + xtalks[11]+ xtalksOffset);

      xt_lr0[t] = (1. - xt_l[0][t] - xt_r[0][t]);
      xt_lr1[t] = (1. - xt_l[1][t] - xt_r[1][t]);
      xt_lr2[t] = (1. - xt_l[2][t] - xt_r[2][t]);
    }
  } else { 
    for ( int t = 0; t < 3; ++t ) {
      xt_l[0][t] = xtalksOffset;
      xt_r[0][t] = xtalksOffset;
      xt_l[1][t] = xtalksOffset; 
      xt_r[1][t] = xtalksOffset; 
      xt_l[2][t] = xtalksOffset; 
      xt_r[2][t] = xtalksOffset; 

      xt_lr0[t] = (1. - xt_l[0][t] - xt_r[0][t]);
      xt_lr1[t] = (1. - xt_l[1][t] - xt_r[1][t]);
      xt_lr2[t] = (1. - xt_l[2][t] - xt_r[2][t]);
    } 
  }
  

  // vector containing noise starts at tmax - 1, and tmax > 3, but....
  int tbin = tmax - 4;

  // .... originally, suppose to have tmax in tbin 4 or 5, but noticed in MTCC lots of 
  // hits with tmax == 3, so let's allow these, and shift down noise matrix by one element...
  // This is a patch because the calibration database doesn't have elements for tbin = 2, 
  // e.g. there is no element e[tmax-1,tmax+1] = e[2,4].

  if (tmax < 4) tbin = 0;    // patch

  // Load in auto-correlation noise matrices
  if ( useCalib ) {
    std::vector<float> nmatrix;
    stripNoiseMatrix_->setNoiseMatrix( globalGainAvg, gains_, noise_ );
    stripNoiseMatrix_->getNoiseMatrix( id, centralStrip, nmatrix);

    for ( int istrip =0; istrip < 3; ++istrip ) {
      a11[istrip] = nmatrix[0+tbin*3+istrip*15];
      a12[istrip] = nmatrix[1+tbin*3+istrip*15];
      a13[istrip] = nmatrix[2+tbin*3+istrip*15];
      a22[istrip] = nmatrix[3+tbin*3+istrip*15];
      a23[istrip] = nmatrix[4+tbin*3+istrip*15];
      a33[istrip] = nmatrix[6+tbin*3+istrip*15];
    }
  } else {
    // FIXME:  NO HARD WIRED VALUES !!!
    for ( int istrip =0; istrip < 3; ++istrip ) {
      a11[istrip] = 10.0;
      a12[istrip] = 0.0;
      a13[istrip] = 0.0;
      a22[istrip] = 10.0;
      a23[istrip] = 0.0;
      a33[istrip] = 10.0;
    }
  }
  
  //---- Set up noise, XTalk matrices 
  setupMatrix();
  Charge = Qsum;
  //---- Calculate the coordinate within the strip and associate uncertainty 
  x_gatti = calculateXonStripPosition(QsumL, QsumC, QsumR, stripWidth);
  xGatti = xCenterStrip + (x_gatti * stripWidth);
  sigma =  calculateXonStripError(QsumL, QsumC, QsumR, stripWidth);
  //chisq = x_gatti;  // chisq is meaningless here
}

/* setupMatrix
 *
 */
void CSCXonStrip_MatchGatti::setupMatrix() {
  //---- a??? and v??[] could be skipped for now...; not used yet

  /*
  double dd, a11t, a12t, a13t, a22t, a23t, a33t;
  double syserr = adcSystematics;
  double xtlk_err = xtalksSystematics;
  // Left strip
  a11t = a11[0] + syserr*syserr * ChargeSignal[0][0]*ChargeSignal[0][0] + xtlk_err*xtlk_err*ChargeSignal[1][0]*ChargeSignal[1][0];
  a12t = a12[0] + syserr*syserr * ChargeSignal[0][0]*ChargeSignal[0][1];
  a13t = a13[0] + syserr*syserr * ChargeSignal[0][0]*ChargeSignal[0][2];
  a22t = a22[0] + syserr*syserr * ChargeSignal[0][1]*ChargeSignal[0][1] + xtlk_err*xtlk_err*ChargeSignal[1][1]*ChargeSignal[1][1];
  a23t = a23[0] + syserr*syserr * ChargeSignal[0][1]*ChargeSignal[0][2];
  a33t = a33[0] + syserr*syserr * ChargeSignal[0][2]*ChargeSignal[0][2] + xtlk_err*xtlk_err*ChargeSignal[1][2]*ChargeSignal[1][2];

  dd     = (a11t*a33t*a22t - a11t*a23t*a23t - a33t*a12t*a12t 
                       + 2.* a12t*a13t*a23t - a13t*a13t*a22t );

  v11[0] = (a33t*a22t - a23t*a23t)/dd;
  v12[0] =-(a33t*a12t - a13t*a23t)/dd;
  v13[0] = (a12t*a23t - a13t*a22t)/dd;
  v22[0] = (a33t*a11t - a13t*a13t)/dd;
  v23[0] =-(a23t*a11t - a12t*a13t)/dd;
  v33[0] = (a22t*a11t - a12t*a12t)/dd;
     
  // Center strip
  a11t = a11[1] + syserr*syserr * ChargeSignal[1][0]*ChargeSignal[1][0] + xtlk_err*xtlk_err*(ChargeSignal[0][0]*ChargeSignal[0][0]+ChargeSignal[2][0]*ChargeSignal[2][0]);
  a12t = a12[1] + syserr*syserr * ChargeSignal[1][0]*ChargeSignal[1][1];
  a13t = a13[1] + syserr*syserr * ChargeSignal[1][0]*ChargeSignal[1][2];
  a22t = a22[1] + syserr*syserr * ChargeSignal[1][1]*ChargeSignal[1][1] + xtlk_err*xtlk_err*(ChargeSignal[0][1]*ChargeSignal[0][1]+ChargeSignal[2][1]*ChargeSignal[2][1]);
  a23t = a23[1] + syserr*syserr * ChargeSignal[1][1]*ChargeSignal[1][2];
  a33t = a33[1] + syserr*syserr * ChargeSignal[1][2]*ChargeSignal[1][2] + xtlk_err*xtlk_err*(ChargeSignal[0][2]*ChargeSignal[0][2]+ChargeSignal[2][2]*ChargeSignal[2][2]);

  dd     = (a11t*a33t*a22t - a11t*a23t*a23t - a33t*a12t*a12t
                       + 2.* a12t*a13t*a23t - a13t*a13t*a22t );

  v11[1] = (a33t*a22t - a23t*a23t)/dd;
  v12[1] =-(a33t*a12t - a13t*a23t)/dd;
  v13[1] = (a12t*a23t - a13t*a22t)/dd;
  v22[1] = (a33t*a11t - a13t*a13t)/dd;
  v23[1] =-(a23t*a11t - a12t*a13t)/dd;
  v33[1] = (a22t*a11t - a12t*a12t)/dd;

  // Right strip
  a11t = a11[2] + syserr*syserr * ChargeSignal[2][0]*ChargeSignal[2][0] + xtlk_err*xtlk_err*ChargeSignal[1][0]*ChargeSignal[1][0];
  a12t = a12[2] + syserr*syserr * ChargeSignal[2][0]*ChargeSignal[2][1];
  a13t = a13[2] + syserr*syserr * ChargeSignal[2][0]*ChargeSignal[2][2];
  a22t = a22[2] + syserr*syserr * ChargeSignal[2][1]*ChargeSignal[2][1] + xtlk_err*xtlk_err*ChargeSignal[1][1]*ChargeSignal[1][1];
  a23t = a23[2] + syserr*syserr * ChargeSignal[2][1]*ChargeSignal[2][2];
  a33t = a33[2] + syserr*syserr * ChargeSignal[2][2]*ChargeSignal[2][2] + xtlk_err*xtlk_err*ChargeSignal[1][2]*ChargeSignal[1][2];

  dd     = (a11t*a33t*a22t - a11t*a23t*a23t - a33t*a12t*a12t
                        +2.* a12t*a13t*a23t - a13t*a13t*a22t );

  v11[2] = (a33t*a22t - a23t*a23t)/dd;
  v12[2] =-(a33t*a12t - a13t*a23t)/dd;
  v13[2] = (a12t*a23t - a13t*a22t)/dd;
  v22[2] = (a33t*a11t - a13t*a13t)/dd;
  v23[2] =-(a23t*a11t - a12t*a13t)/dd;
  v33[2] = (a22t*a11t - a12t*a12t)/dd;
*/
  //---- Find the inverted XTalk matrix and apply it to the charge (3x3)
  //---- Thus the charge before the XTalk is obtained
  HepMatrix XTalks(3,3);
  HepMatrix XTalksInv(3,3);
  int err = 0;
  //---- Qsum is 3 time bins summed; L, C, R - left, central, right strips
  Qsum = QsumL = QsumC = QsumR = 0.;
  double Charge = 0.;
  for(int iTime=0;iTime<3;iTime++){
    XTalksInv(1,1) = XTalks(1,1) = xt_lr0[iTime];
    XTalksInv(1,2) = XTalks(1,2) = xt_l[1][iTime];
    XTalksInv(1,3) = XTalks(1,3) = 0.;
    XTalksInv(2,1) = XTalks(2,1) =  xt_r[0][iTime];
    XTalksInv(2,2) = XTalks(2,2) = xt_lr1[iTime];
    XTalksInv(2,3) = XTalks(2,3) = xt_l[2][iTime];
    XTalksInv(3,1) = XTalks(3,1) = 0.;
    XTalksInv(3,2) = XTalks(3,2) = xt_r[1][iTime];
    XTalksInv(3,3) = XTalks(3,3) = xt_lr2[iTime];
    XTalksInv.invert(err);
    if (err != 0) {
      std::cout<<" Failed to invert XTalks matrix. Bad..."<<std::endl;
    }
    //---- "Charge" is XT-corrected charge
    Charge = ChargeSignal[0][iTime]*XTalksInv(1,1) + ChargeSignal[1][iTime]*XTalksInv(1,2)+ ChargeSignal[2][iTime]*XTalksInv(1,3);
    //---- Negative charge? According to studies - better use 0 charge
    if(Charge<0.){
      Charge = 0.;
    }
    Qsum+=Charge;
    QsumL+=Charge;
    Charge = ChargeSignal[0][iTime]*XTalksInv(2,1) + ChargeSignal[1][iTime] *XTalksInv(2,2)+ ChargeSignal[2][iTime]*XTalksInv(2,3);
    //---- Negative charge? According to studies - better use 0 charge
    if(Charge<0.){
      Charge = 0.;
    }
    Qsum+=Charge;
    QsumC+=Charge;
    Charge = ChargeSignal[0][iTime]*XTalksInv(3,1)  +  ChargeSignal[1][iTime]*XTalksInv(3,2)  + ChargeSignal[2][iTime]*XTalksInv(3,3);
    if(Charge<0.){
      Charge = 0.;
    }
    Qsum+=Charge;
    QsumR+=Charge;
    //std::cout<<" XTalks = "<<XTalks<<std::endl;
    //std::cout<<" XTalksInv = "<<XTalksInv<<std::endl;
  }
  //std::cout<<" QsumL = "<<QsumL<<" QsumC = "<< QsumC<<" QsumR = "<< QsumR<<" Qsum = "<< Qsum<<std::endl; 
  
}


/* initChamberSpecs
 *
 */
void CSCXonStrip_MatchGatti::initChamberSpecs() {
  // Not used directly but these are parameters used for extracting the correction values
  // in coordinate and error estimators

  // Distance between anode and cathode
  h = specs_->anodeCathodeSpacing();
  r = h / stripWidth;

  // Wire spacing
  double wspace = specs_->wireSpacing();

  // Wire radius
  double wradius = specs_->wireRadius();

  // Accepted parameters in Gatti function
  const double parm[5] = {.1989337e-02, -.6901542e-04, .8665786, 154.6177, -.680163e-03 };

  k_3 = ( parm[0]*wspace/h + parm[1] )
      * ( parm[2]*wspace/wradius + parm[3] + parm[4]*(wspace/wradius)*(wspace/wradius) );

  sqrt_k_3 = sqrt( k_3 );
  norm     = r * (0.5 / atan( sqrt_k_3 )); // changed from norm to r * norm
  k_2      = M_PI_2 * ( 1. - sqrt_k_3 /2. );
  k_1      = 0.25 * k_2 * sqrt_k_3 / atan( sqrt_k_3 );
}


void CSCXonStrip_MatchGatti::getCorrectionValues(std::string Estimator){
  HardCodedCorrectionInitialization();
}

double CSCXonStrip_MatchGatti::Estimated2GattiCorrection(double Xestimated, float StripWidth) {
  //---- 14 "nominal" strip widths : 0.3 - 1.6 cm; see getCorrectionValues() 
  //---- Calculate corrections at specific  Xestimated (linear interpolation between points)

  int stripDown = int(10.*StripWidth) - 3;
  int stripUp = stripDown + 1;
  if(stripUp>N_SW-1){
    if(stripUp>N_SW){
      std::cout<<" Is strip width = "<<StripWidth<<" OK?" <<std::endl;
    }
    stripUp = N_SW-1;
  }

  double HalfStripWidth = 0.5;
  //const int Nbins = 501;
  const int Nbins = N_val;
  double CorrToXc = 999.;
  if(stripDown<0){
    CorrToXc = 1;
  }
  else{
    //---- Parametrized Xgatti minus Xestimated differences

    int Xc_bin = -999;
    double deltaStripWidth = 999.;
    double deltaStripWidthUpDown = 999.;
    double DiffToStripWidth = 999.;
    deltaStripWidth = StripWidth - int(StripWidth*10)/10.;
    deltaStripWidthUpDown = 0.1;

    if(fabs(Xestimated)>0.5){
      if(fabs(Xestimated)>1.){
        CorrToXc = 1.;// for now; to be investigated
      }
      else{	 
	//if(fabs(Xestimated)>0.55){
	  //std::cout<<"X position from the estimated position above 0.55 (safty margin)?! "<<std::endl;
	  //CorrToXc = 999.;
	//}
	Xc_bin = int((1.- fabs(Xestimated))/HalfStripWidth * Nbins);
	DiffToStripWidth = Xcorrection[stripUp][Xc_bin]-Xcorrection[stripDown][Xc_bin];
	CorrToXc =  Xcorrection[stripDown][Xc_bin] +
	  (deltaStripWidth/deltaStripWidthUpDown)*DiffToStripWidth ;
	CorrToXc = -CorrToXc;
      }
    }
    else{
      Xc_bin = int((fabs(Xestimated)/HalfStripWidth) * Nbins);
      DiffToStripWidth = Xcorrection[stripUp][Xc_bin]-Xcorrection[stripDown][Xc_bin];
      CorrToXc =  Xcorrection[stripDown][Xc_bin] +
        (deltaStripWidth/deltaStripWidthUpDown)*DiffToStripWidth ;
    }
    if(Xestimated<0.){
      CorrToXc = -CorrToXc;
    }
  }
  
  //std::cout<<" StripWidth = "<<StripWidth<<" Xestimated = "<<Xestimated<<" CorrToXc = "<<CorrToXc<<std::endl;
  return CorrToXc;
}


double CSCXonStrip_MatchGatti::Estimated2Gatti(double Xestimated, float StripWidth) {

 int sign;
 if(Xestimated>0.){
   sign = 1;
 }
 else{
   sign = - 1;
 }
 double Xcorr = Estimated2GattiCorrection(Xestimated, StripWidth);
 double Xgatti = Xestimated + Xcorr;

 return Xgatti;
}

double CSCXonStrip_MatchGatti::XF_error_noise(double L, double C, double R, double noise){

  double min, max;
  if(R>L){
    min = L;
    max = R;
  }
  else{
    min = R;
    max = L;
  }
  //---- Error propagation...
  //---- Names here are fake! Due to technical features
  double dr_L2 = pow(R-L,2);
  double dr_C2 = pow(C-min,2);
  double dr_R2 = pow(C-max,2);
  double error = sqrt(dr_L2 + dr_C2 + dr_R2)*noise/pow(C-min,2)/2;

  return error;
}

double CSCXonStrip_MatchGatti::XF_error_XTasym(double L, double C, double R, double XTasym){

  double min;
  if(R>L){
    min = L;
  }
  else{
    min = R;
  }
  //---- Error propagation
  double dXTL = (pow(C,2)+pow(R,2)-L*R-R*C);
  double dXTR = (pow(C,2)+pow(L,2)-L*R-L*C);
  double dXT = sqrt(pow(dXTL,2) + pow(dXTR,2))/pow((C-min),2)/2;
  double error = dXT * XTasym;

  return error;
}


double CSCXonStrip_MatchGatti::calculateXonStripError(double QsumL, double QsumC, double QsumR, float StripWidth){
  double min;
  if(QsumR>QsumL){
    min = QsumL;
  }
  else{
    min = QsumR;
  }
  
  double XF = (QsumR - QsumL)/(QsumC - min)/2;
  double XF_ErrorNoise = XF_error_noise(QsumL, QsumC, QsumR, NoiseLevel);
  double XF_ErrorXTasym = XF_error_XTasym(QsumL, QsumC, QsumR, XTasymmetry);
  double Xgatti_shift = sqrt( pow( XF_ErrorNoise, 2) + pow( XF_ErrorXTasym, 2)) * 
    (1 + (Estimated2GattiCorrection(XF+0.001,StripWidth) -
	  Estimated2GattiCorrection(XF,StripWidth))*1000.);
  double Xgatti_error =   sqrt( pow( fabs(Xgatti_shift)*StripWidth, 2) + pow(ConstSyst, 2) );
  return  Xgatti_error; 
}

double CSCXonStrip_MatchGatti::calculateXonStripPosition(double QsumL, double QsumC, double QsumR, float StripWidth){

  double  Xestimated = -99.;
  double min;
  if(QsumR>QsumL){
    min = QsumL;
  }
  else{
    min = QsumR;
  }
  //---- This is XF ( X Florida - after the first group that used it)  
  Xestimated = (QsumR - QsumL)/(QsumC - min)/2;
  double Xgatti = Estimated2Gatti(Xestimated, StripWidth);
  return Xgatti;
}


