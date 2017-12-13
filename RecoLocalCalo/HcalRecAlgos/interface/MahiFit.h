#ifndef RecoLocalCalo_HcalRecAlgos_MahiFit_HH
#define RecoLocalCalo_HcalRecAlgos_MahiFit_HH

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/EigenMatrixTypes.h"
#include "DataFormats/HcalRecHit/interface/HBHEChannelInfo.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFunctor.h"

#include "Math/Functor.h"

class MahiFit
{
 public:
  MahiFit();
  ~MahiFit() { };

  void phase1Apply(const HBHEChannelInfo& channelData, float& reconstructedEnergy, float& reconstructedTime, float& chi2);  
  bool doFit(SampleVector amplitudes, std::vector<float> &correctedOutput, int nbx);

  void setParameters(double iTS4Thresh, double chiSqSwitch, bool iApplyTimeSlew, HcalTimeSlew::BiasSetting slewFlavor,
		     double iMeanTime, double iTimeSigmaHPD, double iTimeSigmaSiPM, 
		     const std::vector <int> &iActiveBXs, int iNMaxItersMin, int iNMaxItersNNLS,
		     double iDeltaChiSqThresh, double iNnlsThresh);

  void setPulseShapeTemplate  (const HcalPulseShapes::Shape& ps);
  void resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps);

  typedef BXVector::Index Index;
  const HcalPulseShapes::Shape* currentPulseShape_=nullptr;

 private:

  bool minimize();
  bool onePulseMinimize();
  bool updateCov();
  bool updatePulseShape(double itQ, FullSampleVector &pulseShape, 
			FullSampleVector &pulseDeriv,
			FullSampleMatrix &pulseCov);
  double calculateArrivalTime();
  double calculateChiSq();
  bool nnls();

  void nnlsUnconstrainParameter(Index idxp);
  void nnlsConstrainParameter(Index minratioidx);

  void eigenSolveSubmatrix(PulseMatrix& mat, PulseVector& invec, PulseVector& outvec, unsigned NP);

  double getSiPMDarkCurrent(double darkCurrent, double fcByPE, double lambda);

  //hard coded in initializer
  const unsigned int fullTSSize_;
  const unsigned int fullTSofInterest_;

  // Python-configurables
  float ts4Thresh_; 
  float chiSqSwitch_; 

  bool applyTimeSlew_; 
  HcalTimeSlew::BiasSetting slewFlavor_;

  float meanTime_;
  float timeSigmaHPD_; 
  float timeSigmaSiPM_;

  std::vector <int> activeBXs_;

  int nMaxItersMin_; 
  int nMaxItersNNLS_; 

  float deltaChiSqThresh_; 
  float nnlsThresh_; 

  unsigned int bxSize_;
  int bxOffset_;
  unsigned int bxSizeConf_;
  int bxOffsetConf_;

  //from channelData
  float dt_;
  float darkCurrent_;
  float fcByPe_;

  unsigned int tsSize_;
  unsigned int tsOffset_;

  unsigned int fullTSOffset_;

  //holds active bunch crossings
  BXVector bxs_;  

  BXVector bxsMin_;
  unsigned int nP_;
  double chiSq_;

  //for pulse shapes
  int cntsetPulseShape_;
  std::unique_ptr<FitterFuncs::PulseShapeFunctor> psfPtr_;
  std::unique_ptr<ROOT::Math::Functor> pfunctor_;

  unsigned int nPulseTot_;

  //holds data samples
  SampleVector amplitudes_;

  //holds inverse covariance matrix
  SampleMatrix invCovMat_;

  //holds diagonal noise terms
  SampleVector noiseTerms_;
  //holds constant pedestal constraint
  double pedConstraint_;
  
  //holds full covariance matrix for a pulse shape 
  //varied in time
  std::array<FullSampleMatrix, MaxPVSize> pulseCovArray_;

  //holds full pulse shape template
  std::array<FullSampleVector, MaxPVSize> pulseShapeArray_;

  //holds full pulse shape derivatives
  std::array<FullSampleVector, MaxPVSize> pulseDerivArray_;

  //holders for calculating pulse shape & covariance matrices
  std::array<double, HcalConst::maxSamples> pulseN_;
  std::array<double, HcalConst::maxSamples> pulseM_;
  std::array<double, HcalConst::maxSamples> pulseP_;

  //holds matrix of pulse shape templates for each BX
  SamplePulseMatrix pulseMat_;

  //holds matrix of pulse shape derivatives for each BX
  SamplePulseMatrix pulseDerivMat_;

  //holds residual vector
  PulseVector residuals_;

  //for FNNLS algorithm
  PulseVector ampVec_;
  PulseVector ampVecMin_;
  PulseVector errVec_;
  PulseVector ampvecpermtest_;

  SamplePulseMatrix invcovp_;
  PulseMatrix aTaMat_; // A-transpose A (matrix)
  PulseVector aTbVec_; // A-transpose b (vector)
  PulseVector wVec_; // w (vector)
  PulseVector updateWork_; // w (vector)

  SampleDecompLLT covDecomp_;
  SampleMatrix covDecompLinv_;
  PulseMatrix topleft_work_;
  PulseDecompLDLT pulseDecomp_;

}; 
#endif
