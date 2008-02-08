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
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <map>  
#include <vector>

class CSCLayer;
class CSCChamberSpecs;
class CSCFindPeakTime;
class CSCDBGains;
class CSCDBCrosstalk;
class CSCDBNoiseMatrix;
class CSCStripCrosstalk;
class CSCStripNoiseMatrix;

class CSCXonStrip_MatchGatti
{
 public:

  explicit CSCXonStrip_MatchGatti(const edm::ParameterSet& ps);

  ~CSCXonStrip_MatchGatti();
  

  // Member functions
  
  /// Returns fitted local x position and its estimated error.
  void findXOnStrip( const CSCDetId& id, const CSCLayer* layer, const CSCStripHit& stripHit,
                     int centralStrip, float& xCentroid, float& stripWidth,
                     double& xGatti, float& tpeak, double& sigma, float& chisq, float & Charge );

  /// Use specs to setup Gatti parameters
  void initChamberSpecs();                       

  /// Set matrix for XT corrections and noise
  void setupMatrix();


  /// Load in x-Talks and Noise Matrix
  void setCalibration( float GlobalGainAvg,
                       const CSCDBGains* gains,
                       const CSCDBCrosstalk* xtalk,
                       const CSCDBNoiseMatrix* noise ) {
    globalGainAvg = GlobalGainAvg;
    gains_ = gains;
    xtalk_ = xtalk;
    noise_ = noise;
  }

  // "Match Gatti" calculations
  double calculateXonStripError(double QsumL, double QsumC, double QsumR, float StripWidth);
  double calculateXonStripPosition(double QsumL, double QsumC, double QsumR, float StripWidth);
  
 private:


  double h;                                     // This is the distance between strip and wire planes
  float stripWidth;
  double r;                                     // This is the ratio h/stripwidth
  
  double k_1, k_2, k_3, sqrt_k_3, norm;         // See equation above for description
    
  // The charge (3x3); [1][1] is the maximum 
  float ChargeSignal[3][3];                                // 3x3 data array for gatti fit

  /// x-talks  0 = left, 1 = middle, 2 = right ; and then second [] is for time bin tmax-1, tmax, tmax+1
  float xt_l[3][3], xt_r[3][3];
  float xt_lr0[3], xt_lr1[3], xt_lr2[3];

  /// Store elements of matrices for chi^2 computation: 0 = left, 1 = middle, 2 = right
  float v11[3], v12[3], v13[3], v22[3], v23[3], v33[3];

  /// Store elements of auto-correlation matrices:      0 = left, 1 = middle, 2 = right
  float a11[3], a12[3], a13[3], a22[3], a23[3], a33[3];


  // The corrected coordinate is supposed to match the Gatti coordinate in the ideal case
  float x_gatti ;

  // Store chamber specs
  const CSCChamberSpecs* specs_;

  // Store XT-corrected charges - 3x3 sum; Left, Central, Right charges (3 time-bins summed) 

  double Qsum, QsumL, QsumC, QsumR;

  // Parameter settings from config file
  bool debug;
  bool useCalib;
  //bool isData;
  bool use3TimeBins;
  double adcSystematics;
  float xtalksOffset;
  float xtalksSystematics;

  /* Cache calibrations for current event
   *
   */
  float globalGainAvg;
  const CSCDBGains*       gains_;
  const CSCDBCrosstalk*   xtalk_;
  const CSCDBNoiseMatrix* noise_;

  // other classes used
  CSCStripCrosstalk*       stripCrosstalk_;
  CSCStripNoiseMatrix*     stripNoiseMatrix_;
  CSCFindPeakTime*         peakTimeFinder_;  

  // some variables and funfctions to use
  // L, C, R : left, central and right strip charges
  double XF_error_noise(double L, double C, double R, double noise);
  double XF_error_XTasym(double L, double C, double R, double XTasym);

  double Estimated2Gatti(double Xcentroid, float StripWidth);
  double Estimated2GattiCorrection(double Xcentroid, float StripWidth);

  void getCorrectionValues(std::string Estimator);
  void HardCodedCorrectionInitialization();

  static const int N_SW = 14;
  static const int N_val = 501;
  //std::vector <std::vector <float> > Xcorrection(N_SW, std::vector <float> (N_val));
  float Xcorrection[N_SW][N_val];
  float XcentrVal[N_val];
  float NoiseLevel;
  float XTasymmetry;
  float ConstSyst;
}; 

#endif
