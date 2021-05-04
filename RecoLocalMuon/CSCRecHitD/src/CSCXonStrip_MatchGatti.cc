// This is CSCXonStrip_MatchGatti.cc

//---- Large part is copied from RecHitB
//---- author: Stoyan Stoynev - NU

#include <RecoLocalMuon/CSCRecHitD/src/CSCXonStrip_MatchGatti.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>

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
//#include <iomanip.h>
//#include <iomanip>

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

CSCXonStrip_MatchGatti::CSCXonStrip_MatchGatti(const edm::ParameterSet& ps) : recoConditions_(nullptr) {
  useCalib = ps.getParameter<bool>("CSCUseCalibrations");
  xtalksOffset = ps.getParameter<double>("CSCStripxtalksOffset");
  noise_level_ME1a = ps.getParameter<double>("NoiseLevel_ME1a");
  xt_asymmetry_ME1a = ps.getParameter<double>("XTasymmetry_ME1a");
  const_syst_ME1a = ps.getParameter<double>("ConstSyst_ME1a");
  noise_level_ME1b = ps.getParameter<double>("NoiseLevel_ME1b");
  xt_asymmetry_ME1b = ps.getParameter<double>("XTasymmetry_ME1b");
  const_syst_ME1b = ps.getParameter<double>("ConstSyst_ME1b");
  noise_level_ME12 = ps.getParameter<double>("NoiseLevel_ME12");
  xt_asymmetry_ME12 = ps.getParameter<double>("XTasymmetry_ME12");
  const_syst_ME12 = ps.getParameter<double>("ConstSyst_ME12");
  noise_level_ME13 = ps.getParameter<double>("NoiseLevel_ME13");
  xt_asymmetry_ME13 = ps.getParameter<double>("XTasymmetry_ME13");
  const_syst_ME13 = ps.getParameter<double>("ConstSyst_ME13");
  noise_level_ME21 = ps.getParameter<double>("NoiseLevel_ME21");
  xt_asymmetry_ME21 = ps.getParameter<double>("XTasymmetry_ME21");
  const_syst_ME21 = ps.getParameter<double>("ConstSyst_ME21");
  noise_level_ME22 = ps.getParameter<double>("NoiseLevel_ME22");
  xt_asymmetry_ME22 = ps.getParameter<double>("XTasymmetry_ME22");
  const_syst_ME22 = ps.getParameter<double>("ConstSyst_ME22");
  noise_level_ME31 = ps.getParameter<double>("NoiseLevel_ME31");
  xt_asymmetry_ME31 = ps.getParameter<double>("XTasymmetry_ME31");
  const_syst_ME31 = ps.getParameter<double>("ConstSyst_ME31");
  noise_level_ME32 = ps.getParameter<double>("NoiseLevel_ME32");
  xt_asymmetry_ME32 = ps.getParameter<double>("XTasymmetry_ME32");
  const_syst_ME32 = ps.getParameter<double>("ConstSyst_ME32");
  noise_level_ME41 = ps.getParameter<double>("NoiseLevel_ME41");
  xt_asymmetry_ME41 = ps.getParameter<double>("XTasymmetry_ME41");
  const_syst_ME41 = ps.getParameter<double>("ConstSyst_ME41");

  getCorrectionValues("StringCurrentlyNotUsed");
}

CSCXonStrip_MatchGatti::~CSCXonStrip_MatchGatti() {}

/* findPosition
 *
 */
void CSCXonStrip_MatchGatti::findXOnStrip(const CSCDetId& id,
                                          const CSCLayer* layer,
                                          const CSCStripHit& stripHit,
                                          int centralStrip,
                                          float& xWithinChamber,
                                          float& sWidth,
                                          const float& tpeak,
                                          float& xWithinStrip,
                                          float& sigma,
                                          int& quality_flag) {
  quality_flag = 0;
  // Initialize Gatti parameters using chamberSpecs
  // Cache specs_ info for ease of access
  specs_ = layer->chamber()->specs();
  stripWidth = sWidth;
  //initChamberSpecs();
  // Initialize output parameters
  xWithinStrip = xWithinChamber;

  CSCStripHit::ChannelContainer const& strips = stripHit.strips();
  int nStrips = strips.size();
  int centStrip = nStrips / 2 + 1;
  std::vector<float> const& adcs = stripHit.s_adc();
  int tmax = stripHit.tmax();

  //// Fit peaking time only if using calibrations
  //float t_peak = tpeak;
  //float t_zero = 0.;
  //float adc[4];
  //
  //if ( useCalib ) {
  //
  //  for ( int t = 0; t < 4; ++t ) {
  //    int k  = t + 4 * (centStrip-1);
  //    adc[t] = adcs[k];
  //  }
  //
  //  // t_peak from peak finder is now 'absolute' i.e. in ns from start of sca time bin 0
  //  t_peak = peakTimeFinder_->peakTime( tmax, adc, t_peak );
  //  // Just for completeness, the start time of the pulse is 133 ns earlier, according to Stan :)
  //  t_zero = t_peak - 133.;
  //  // and reset tpeak since that's the way it gets passed out of this function (Argh!)
  //  tpeak = t_peak;
  //  LogTrace("CSCRecHit|CSCXonStrip_MatchGatti") << "CSCXonStrip_MatchGatti: " <<
  //     id << " strip=" << centralStrip << ", t_zero=" << t_zero << ", t_peak=" << t_peak;
  //}

  //---- fill the charge matrix (3x3)
  float adc[4];
  int j = 0;
  for (int i = 1; i <= nStrips; ++i) {
    if (i > (centStrip - 2) && i < (centStrip + 2)) {
      std::vector<float> adcsFit;
      for (int t = 0; t < 4; ++t) {
        int k = t + 4 * (i - 1);
        adc[t] = adcs[k];
        if (t < 3)
          chargeSignal[j][t] = adc[t];
      }
      j++;
    }
  }

  // Load in x-talks:

  if (useCalib) {
    std::vector<float> xtalks;
    recoConditions_->crossTalk(id, centralStrip, xtalks);
    float dt = 50.f * tmax - tpeak;
    // XTalks; l,r are for left, right XTalk; lr0,1,2 are for what charge "remains" in the strip
    for (int t = 0; t < 3; ++t) {
      xt_l[0][t] = xtalks[0] * (50.f * (t - 1) + dt) + xtalks[1] + xtalksOffset;
      xt_r[0][t] = xtalks[2] * (50.f * (t - 1) + dt) + xtalks[3] + xtalksOffset;
      xt_l[1][t] = xtalks[4] * (50.f * (t - 1) + dt) + xtalks[5] + xtalksOffset;
      xt_r[1][t] = xtalks[6] * (50.f * (t - 1) + dt) + xtalks[7] + xtalksOffset;
      xt_l[2][t] = xtalks[8] * (50.f * (t - 1) + dt) + xtalks[9] + xtalksOffset;
      xt_r[2][t] = xtalks[10] * (50.f * (t - 1) + dt) + xtalks[11] + xtalksOffset;

      xt_lr0[t] = (1. - xt_l[0][t] - xt_r[0][t]);
      xt_lr1[t] = (1. - xt_l[1][t] - xt_r[1][t]);
      xt_lr2[t] = (1. - xt_l[2][t] - xt_r[2][t]);
    }
  } else {
    for (int t = 0; t < 3; ++t) {
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

  if (tmax < 4)
    tbin = 0;  // patch

  // Load in auto-correlation noise matrices
  if (useCalib) {
    std::vector<float> nmatrix;
    recoConditions_->noiseMatrix(id, centralStrip, nmatrix);
    for (int istrip = 0; istrip < 3; ++istrip) {
      a11[istrip] = nmatrix[0 + tbin * 3 + istrip * 15];
      a12[istrip] = nmatrix[1 + tbin * 3 + istrip * 15];
      a13[istrip] = nmatrix[2 + tbin * 3 + istrip * 15];
      a22[istrip] = nmatrix[3 + tbin * 3 + istrip * 15];
      a23[istrip] = nmatrix[4 + tbin * 3 + istrip * 15];
      a33[istrip] = nmatrix[6 + tbin * 3 + istrip * 15];
    }
  } else {
    // FIXME:  NO HARD WIRED VALUES !!!
    for (int istrip = 0; istrip < 3; ++istrip) {
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
  //---- Calculate the coordinate within the strip and associate uncertainty

  static const std::string ME1a("ME1/a");
  static const std::string ME1b("ME1/b");

  bool ME1_1 = (ME1a == specs_->chamberTypeName() || ME1b == specs_->chamberTypeName());

  // due to cross talks and 3 time bin sum it is in principe possible that the center strip is not the maximum strip
  // in some cases the consequences could be quite extreme
  // take some measures against the extreme cases
  bool peakMismatch = false;
  std::vector<float> charges(3);
  charges[0] = q_sumL;
  charges[1] = q_sumC;
  charges[2] = q_sumR;
  int min_index = min_element(charges.begin(), charges.end()) - charges.begin();
  int max_index = max_element(charges.begin(), charges.end()) - charges.begin();
  if (1 != max_index && (1 == min_index ||
                         // the condition below means that if the initial position estimate within strip (|xF|)
                         // is above  1.1/2 = 0.55 "strip widths" peakMismatch is set to true (so special case);
                         // in normal cases |xF|<=0.5 (0.5 is at the edge of the "central" strip)
                         (charges[max_index] - charges[min_index]) > 1.1 * (q_sumC - charges[min_index]))) {
    peakMismatch = true;
    switch (max_index) {
      case 0:
        xWithinStrip = -1;
        break;
      case 2:
        xWithinStrip = 1;
        break;
      default:
        // should be an error message here
        xWithinStrip = 0;  // in case?
        break;
    }
  }
  // we don't have the needed information (it's similar to the "edge" strip case)
  //else if(stripHit.isNearDeadStrip()){
  else if (stripHit.deadStrip() > 0) {
    xWithinStrip = 0;
  } else {
    //
    xWithinStrip = float(calculateXonStripPosition(stripWidth, ME1_1));
  }
  xWithinChamber = xWithinChamber + (xWithinStrip * stripWidth);
  if (peakMismatch) {
    sigma = stripWidth / std::sqrt(12.f);
  } else {
    //---- error estimation
    int factorStripWidth = int(std::sqrt(stripWidth / 0.38f));
    int maxConsecutiveStrips = 8;
    if (factorStripWidth) {
      maxConsecutiveStrips /= factorStripWidth;
    }
    maxConsecutiveStrips++;

    struct ChamberTypes {
      std::map<std::string, int> chamberTypes;
      int operator()(std::string const& s) const {
        auto p = chamberTypes.find(s);
        return p != chamberTypes.end() ? (*p).second : 0;
      }
      ChamberTypes() {
        chamberTypes["ME1/a"] = 1;
        chamberTypes["ME1/b"] = 2;
        chamberTypes["ME1/2"] = 3;
        chamberTypes["ME1/3"] = 4;
        chamberTypes["ME2/1"] = 5;
        chamberTypes["ME2/2"] = 6;
        chamberTypes["ME3/1"] = 7;
        chamberTypes["ME3/2"] = 8;
        chamberTypes["ME4/1"] = 9;
        chamberTypes["ME4/2"] = 8;
      }
    };
    static const ChamberTypes chamberTypes;

    switch (chamberTypes(specs_->chamberTypeName())) {
      case 1:
        noise_level = noise_level_ME1a;
        xt_asymmetry = xt_asymmetry_ME1a;
        const_syst = const_syst_ME1a;
        break;

      case 2:
        noise_level = noise_level_ME1b;
        xt_asymmetry = xt_asymmetry_ME1b;
        const_syst = const_syst_ME1b;
        break;

      case 3:
        noise_level = noise_level_ME12;
        xt_asymmetry = xt_asymmetry_ME12;
        const_syst = const_syst_ME12;
        break;

      case 4:
        noise_level = noise_level_ME13;
        xt_asymmetry = xt_asymmetry_ME13;
        const_syst = const_syst_ME13;
        break;

      case 5:
        noise_level = noise_level_ME21;
        xt_asymmetry = xt_asymmetry_ME21;
        const_syst = const_syst_ME21;
        break;

      case 6:
        noise_level = noise_level_ME22;
        xt_asymmetry = xt_asymmetry_ME22;
        const_syst = const_syst_ME22;
        break;

      case 7:
        noise_level = noise_level_ME31;
        xt_asymmetry = xt_asymmetry_ME31;
        const_syst = const_syst_ME31;
        break;

      case 8:
        noise_level = noise_level_ME32;
        xt_asymmetry = xt_asymmetry_ME32;
        const_syst = const_syst_ME32;
        break;

      case 9:
        noise_level = noise_level_ME41;
        xt_asymmetry = xt_asymmetry_ME41;
        const_syst = const_syst_ME41;
        break;

      default:
        noise_level = noise_level_ME22;
        xt_asymmetry = xt_asymmetry_ME22;
        const_syst = const_syst_ME22;
    }
    if (0 == stripHit.deadStrip() && stripHit.numberOfConsecutiveStrips() < maxConsecutiveStrips &&
        std::abs(stripHit.closestMaximum()) > maxConsecutiveStrips / 2) {
      sigma = float(calculateXonStripError(stripWidth, ME1_1));
    } else {  //---- near dead strip or too close maxima or too wide strip cluster
      sigma = stripWidth / std::sqrt(12.f);
    }
  }
  quality_flag = 1;
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
  CLHEP::HepMatrix cross_talks_inv(3, 3);
  int err = 0;
  //---- q_sum is 3 time bins summed; L, C, R - left, central, right strips
  q_sum = q_sumL = q_sumC = q_sumR = 0.;
  double charge = 0.;
  for (int iTime = 0; iTime < 3; iTime++) {
    cross_talks_inv(1, 1) = xt_lr0[iTime];
    cross_talks_inv(1, 2) = xt_l[1][iTime];
    cross_talks_inv(1, 3) = 0.;
    cross_talks_inv(2, 1) = xt_r[0][iTime];
    cross_talks_inv(2, 2) = xt_lr1[iTime];
    cross_talks_inv(2, 3) = xt_l[2][iTime];
    cross_talks_inv(3, 1) = 0.;
    cross_talks_inv(3, 2) = xt_r[1][iTime];
    cross_talks_inv(3, 3) = xt_lr2[iTime];
    cross_talks_inv.invert(err);
    if (err != 0) {
      edm::LogWarning("FailedXTalkiInversionNoCrosstalkCorrection")
          << "Failed to invert XTalks matrix. No cross-talk correction for this rechit.";
      //edm::LogError("CSCRecHit") << "Failed to invert XTalks matrix. No cross-talk correction for this rechit.";
      return;
    }
    //---- "charge" is XT-corrected charge
    charge = chargeSignal[0][iTime] * cross_talks_inv(1, 1) + chargeSignal[1][iTime] * cross_talks_inv(1, 2) +
             chargeSignal[2][iTime] * cross_talks_inv(1, 3);
    //---- Negative charge? According to studies (and logic) - better use 0 charge
    //----- Later studies suggest that this only do harm. I am still worried about
    // charges of -50 ADC and below (0.5% of the cases) but let see
    //if(charge<0.){
    //charge = 0.;
    //}
    q_sum += charge;
    q_sumL += charge;
    charge = chargeSignal[0][iTime] * cross_talks_inv(2, 1) + chargeSignal[1][iTime] * cross_talks_inv(2, 2) +
             chargeSignal[2][iTime] * cross_talks_inv(2, 3);
    //if(charge<0.){
    //charge = 0.;
    //}
    q_sum += charge;
    q_sumC += charge;
    charge = chargeSignal[0][iTime] * cross_talks_inv(3, 1) + chargeSignal[1][iTime] * cross_talks_inv(3, 2) +
             chargeSignal[2][iTime] * cross_talks_inv(3, 3);
    //if(charge<0.){
    //charge = 0.;
    //}
    q_sum += charge;
    q_sumR += charge;
  }
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
  const double parm[5] = {.1989337e-02, -.6901542e-04, .8665786, 154.6177, -.680163e-03};

  k_3 = (parm[0] * wspace / h + parm[1]) *
        (parm[2] * wspace / wradius + parm[3] + parm[4] * (wspace / wradius) * (wspace / wradius));

  sqrt_k_3 = std::sqrt(k_3);
  norm = r * (0.5 / std::atan(sqrt_k_3));  // changed from norm to r * norm
  k_2 = M_PI_2 * (1. - sqrt_k_3 / 2.);
  k_1 = 0.25 * k_2 * sqrt_k_3 / std::atan(sqrt_k_3);
}

void CSCXonStrip_MatchGatti::getCorrectionValues(std::string estimator) { hardcodedCorrectionInitialization(); }

double CSCXonStrip_MatchGatti::estimated2GattiCorrection(double x_estimated, float stripWidth, bool ME1_1) {
  //---- 11 "nominal" strip widths : 0.6 - 1.6 cm; for ME1_1 just 6 "nominal" strip widths : 0.3 - 0.8 cm; see HardCodedCorrectionInitialization()
  //---- Calculate corrections at specific  Xestimated (linear interpolation between points)
  int n_SW;
  int min_SW;
  if (ME1_1) {
    n_SW = n_SW_ME1_1;
    min_SW = 3;  // min SW calculated is 0.3 cm
  } else {
    n_SW = n_SW_noME1_1;
    min_SW = 6;  // min SW calculated is 0.6 cm
  }
  int stripDown = int(10. * stripWidth) - min_SW;  // 0 is at min strip width calculated
  int stripUp = stripDown + 1;
  if (stripUp > n_SW - 1) {
    //---- to be checked...
    //if(stripUp>n_SW){
    //std::cout<<" Is strip width = "<<stripWidth<<" OK?" <<std::endl;
    //}
    stripUp = n_SW - 1;
  }

  double half_strip_width = 0.5;
  //const int Nbins = 501;
  const int n_bins = n_val;
  double corr_2_xestim = 999.;
  if (stripDown < 0) {
    corr_2_xestim = 1;
  } else {
    //---- Parametrized x_gatti minus x_estimated differences

    int xestim_bin = -999;
    double delta_strip_width = 999.;
    double delta_strip_widthUpDown = 999.;
    double diff_2_strip_width = 999.;
    delta_strip_width = stripWidth - int(stripWidth * 10) / 10.;
    delta_strip_widthUpDown = 0.1;

    if (std::abs(x_estimated) > 0.5) {
      if (std::abs(x_estimated) > 1.) {
        corr_2_xestim = 1.;  // for now; to be investigated
      } else {
        //if(std::abs(Xestimated)>0.55){
        //std::cout<<"X position from the estimated position above 0.55 (safty margin)?! "<<std::endl;
        //CorrToXc = 999.;
        //}
        xestim_bin = int((1. - std::abs(x_estimated)) / half_strip_width * n_bins);
        if (ME1_1) {
          diff_2_strip_width = x_correction_ME1_1[stripUp][xestim_bin] - x_correction_ME1_1[stripDown][xestim_bin];
          corr_2_xestim = x_correction_ME1_1[stripDown][xestim_bin] +
                          (delta_strip_width / delta_strip_widthUpDown) * diff_2_strip_width;
        } else {
          diff_2_strip_width = x_correction_noME1_1[stripUp][xestim_bin] - x_correction_noME1_1[stripDown][xestim_bin];
          corr_2_xestim = x_correction_noME1_1[stripDown][xestim_bin] +
                          (delta_strip_width / delta_strip_widthUpDown) * diff_2_strip_width;
        }
        corr_2_xestim = -corr_2_xestim;
      }
    } else {
      xestim_bin = int((std::abs(x_estimated) / half_strip_width) * n_bins);
      if (ME1_1) {
        diff_2_strip_width = x_correction_ME1_1[stripUp][xestim_bin] - x_correction_ME1_1[stripDown][xestim_bin];
        corr_2_xestim = x_correction_ME1_1[stripDown][xestim_bin] +
                        (delta_strip_width / delta_strip_widthUpDown) * diff_2_strip_width;
      } else {
        diff_2_strip_width = x_correction_noME1_1[stripUp][xestim_bin] - x_correction_noME1_1[stripDown][xestim_bin];
        corr_2_xestim = x_correction_noME1_1[stripDown][xestim_bin] +
                        (delta_strip_width / delta_strip_widthUpDown) * diff_2_strip_width;
      }
    }
    if (x_estimated < 0.) {
      corr_2_xestim = -corr_2_xestim;
    }
  }

  return corr_2_xestim;
}

double CSCXonStrip_MatchGatti::estimated2Gatti(double x_estimated, float stripWidth, bool ME1_1) {
  double x_corr = estimated2GattiCorrection(x_estimated, stripWidth, ME1_1);
  double x_gatti = x_estimated + x_corr;

  return x_gatti;
}

double CSCXonStrip_MatchGatti::xfError_Noise(double noise) {
  double min, max;
  if (q_sumR > q_sumL) {
    min = q_sumL;
    max = q_sumR;
  } else {
    min = q_sumR;
    max = q_sumL;
  }
  //---- Error propagation...
  //---- Names here are fake! Due to technical features
  double dr_L2 = pow(q_sumR - q_sumL, 2);
  double dr_C2 = pow(q_sumC - min, 2);
  double dr_R2 = pow(q_sumC - max, 2);
  double error = std::sqrt(dr_L2 + dr_C2 + dr_R2) * noise / std::pow(q_sumC - min, 2) / 2;

  return error;
}

double CSCXonStrip_MatchGatti::xfError_XTasym(double xt_asym) {
  double min;
  if (q_sumR > q_sumL) {
    min = q_sumL;
  } else {
    min = q_sumR;
  }
  //---- Error propagation
  double dXTL = (std::pow(q_sumC, 2) + std::pow(q_sumR, 2) - q_sumL * q_sumR - q_sumR * q_sumC);
  double dXTR = (std::pow(q_sumC, 2) + std::pow(q_sumL, 2) - q_sumL * q_sumR - q_sumL * q_sumC);
  double dXT = std::sqrt(std::pow(dXTL, 2) + std::pow(dXTR, 2)) / std::pow((q_sumC - min), 2) / 2;
  double error = dXT * xt_asym;

  return error;
}

double CSCXonStrip_MatchGatti::calculateXonStripError(float stripWidth, bool ME1_1) {
  double min;
  if (q_sumR > q_sumL) {
    min = q_sumL;
  } else {
    min = q_sumR;
  }

  double xf = (q_sumR - q_sumL) / (q_sumC - min) / 2;
  double xf_ErrorNoise = xfError_Noise(noise_level);
  double xf_ErrorXTasym = xfError_XTasym(xt_asymmetry);
  // x_G = x_F + correction_functon(x_F)
  // as these are correlated the error should be simply d(x_G) = |d(x_F)| + [correction_functon(x_F+|d(x_F)|) - correction_functon(x_F)]
  double d_xf = std::sqrt(std::pow(xf_ErrorNoise, 2) + std::pow(xf_ErrorXTasym, 2));
  double d_corr =
      estimated2GattiCorrection(xf + d_xf, stripWidth, ME1_1) - estimated2GattiCorrection(xf, stripWidth, ME1_1);
  double x_shift = d_xf + d_corr;
  //  double x_shift = sqrt( pow( xf_ErrorNoise, 2) + pow( xf_ErrorXTasym, 2)) *
  //(1 + (estimated2GattiCorrection(xf+0.001, stripWidth, ME1_1) -
  //  estimated2GattiCorrection(xf, stripWidth, ME1_1))*1000.);
  double x_error = std::sqrt(std::pow(std::abs(x_shift) * stripWidth, 2) + std::pow(const_syst, 2));
  return x_error;
}

double CSCXonStrip_MatchGatti::calculateXonStripPosition(float stripWidth, bool ME1_1) {
  double x_estimated = -99.;
  double min;
  if (q_sumR > q_sumL) {
    min = q_sumL;
  } else {
    min = q_sumR;
  }
  //---- This is XF ( X Florida - after the first group that used it)
  x_estimated = (q_sumR - q_sumL) / (q_sumC - min) / 2;
  double x_gatti = estimated2Gatti(x_estimated, stripWidth, ME1_1);
  return x_gatti;
}

// Define space for statics
const int CSCXonStrip_MatchGatti::n_val;
const int CSCXonStrip_MatchGatti::n_SW_noME1_1;
const int CSCXonStrip_MatchGatti::n_SW_ME1_1;
