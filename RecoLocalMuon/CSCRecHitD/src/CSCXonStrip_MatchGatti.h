#ifndef CSCRecHitD_CSCXonStrip_MatchGatti_h
#define CSCRecHitD_CSCXonStrip_MatchGatti_h
//---- Large part taken from RecHitB

/** \class CSCXonStrip_MatchGatti
 *
 * When having both strip and wire hit in a layer, use Gatti "matching" to
 * calculate position ond error of strip hit.  
 *
 * \author Stoyan Stoynev - NU 
 *
 */

#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <map>  
#include <vector>

class CSCLayer;
class CSCChamberSpecs;

class CSCXonStrip_MatchGatti
{
 public:

  explicit CSCXonStrip_MatchGatti(const edm::ParameterSet& ps);

  ~CSCXonStrip_MatchGatti();
  

  // Member functions
  
  /// Returns fitted local x position and its estimated error.
  void findXOnStrip( const CSCDetId& id, const CSCLayer* layer, const CSCStripHit& stripHit,
                     int centralStrip, float& xWithinChamber, float& stripWidth,
                     const float& tpeak, float& xWithinStrip, float& sigma, int & quality_flag);

  /// Use specs to setup Gatti parameters
  void initChamberSpecs();                       

  /// Set matrix for XT corrections and noise
  void setupMatrix();

  /// Cache pointer to conditions data
  void setConditions( const CSCRecoConditions* reco ) {
    recoConditions_ = reco;
  }

 private:

  // No copying of this class
  CSCXonStrip_MatchGatti( const CSCXonStrip_MatchGatti& );
  CSCXonStrip_MatchGatti& operator=( const CSCXonStrip_MatchGatti& );

  double h;                                     // This is the distance between strip and wire planes
  float stripWidth;
  double r;                                     // This is the ratio h/stripwidth
  
  double k_1, k_2, k_3, sqrt_k_3, norm;         // See equation above for description
    
  // The charge (3x3); [1][1] is the maximum 
  float chargeSignal[3][3];                                // 3x3 data array for gatti fit

  /// x-talks  0 = left, 1 = middle, 2 = right ; and then second [] is for time bin tmax-1, tmax, tmax+1
  float xt_l[3][3], xt_r[3][3];
  float xt_lr0[3], xt_lr1[3], xt_lr2[3];

  /// Store elements of matrices for chi^2 computation: 0 = left, 1 = middle, 2 = right
  float v11[3], v12[3], v13[3], v22[3], v23[3], v33[3];

  /// Store elements of auto-correlation matrices:      0 = left, 1 = middle, 2 = right
  float a11[3], a12[3], a13[3], a22[3], a23[3], a33[3];


  // Store chamber specs
  const CSCChamberSpecs* specs_;

  // Store XT-corrected charges - 3x3 sum; Left, Central, Right charges (3 time-bins summed) 

  double q_sum, q_sumL, q_sumC, q_sumR;

  // Parameter settings from config file
  bool useCalib;
  bool use3TimeBins;
  float xtalksOffset;

  // Cache pointer to conditions for current event
  const CSCRecoConditions* recoConditions_;

  // some variables and functions to use

  // "Match Gatti" calculations
  double calculateXonStripError(float stripWidth, bool ME1_1);
  double calculateXonStripPosition(float stripWidth, bool ME1_1);
  double xfError_Noise(double noise);
  double xfError_XTasym(double XTasym);

  double estimated2Gatti(double Xestimated, float StripWidth, bool ME1_1);
  double estimated2GattiCorrection(double Xestimated, float StripWidth, bool ME1_1);

  void getCorrectionValues(std::string Estimator);
  void hardcodedCorrectionInitialization();

  static const int n_SW_noME1_1 = 11;
  static const int n_SW_ME1_1 = 6;
  static const int n_val = 501;
  //std::vector <std::vector <float> > Xcorrection(N_SW, std::vector <float> (N_val));
  float x_correction_noME1_1[n_SW_noME1_1][n_val];
  float x_correction_ME1_1[n_SW_ME1_1][n_val];
  float x_centralVal[n_val];

  float noise_level;
  float xt_asymmetry;
  float const_syst;

  float noise_level_ME1a;
  float xt_asymmetry_ME1a;
  float const_syst_ME1a;
  float noise_level_ME1b;
  float xt_asymmetry_ME1b;
  float const_syst_ME1b;
  float noise_level_ME12;
  float xt_asymmetry_ME12;
  float const_syst_ME12;
  float noise_level_ME13;
  float xt_asymmetry_ME13;
  float const_syst_ME13;
  float noise_level_ME21;
  float xt_asymmetry_ME21;
  float const_syst_ME21;
  float noise_level_ME22;
  float xt_asymmetry_ME22;
  float const_syst_ME22;
  float noise_level_ME31;
  float xt_asymmetry_ME31;
  float const_syst_ME31;
  float noise_level_ME32;
  float xt_asymmetry_ME32;
  float const_syst_ME32;
  float noise_level_ME41;
  float xt_asymmetry_ME41;
  float const_syst_ME41;
}; 

#endif
