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

#include <Math/Functor.h>

struct nnlsWorkspace {

  unsigned int nPulseTot;
  //unsigned int bxSize;
  int bxOffset;

  //holds active bunch crossings
  BXVector bxs;  

  //holds data samples
  SampleVector amplitudes;

  //holds inverse covariance matrix
  SampleMatrix invCovMat;

  //holds diagonal noise terms
  SampleVector noiseTerms;
  
  //holds full covariance matrix for a pulse shape 
  //varied in time
  std::array<FullSampleMatrix, MaxPVSize> pulseCovArray;

  //holds full pulse shape template
  std::array<FullSampleVector, MaxPVSize> pulseShapeArray;

  //holds full pulse shape derivatives
  std::array<FullSampleVector, MaxPVSize> pulseDerivArray;

  //holders for calculating pulse shape & covariance matrices
  std::array<double, MaxSVSize> pulseN;
  std::array<double, MaxSVSize> pulseM;
  std::array<double, MaxSVSize> pulseP;

  //holds matrix of pulse shape templates for each BX
  SamplePulseMatrix pulseMat;

  //holds matrix of pulse shape derivatives for each BX
  SamplePulseMatrix pulseDerivMat;

  //holds residual vector
  PulseVector residuals;

  //for FNNLS algorithm
  PulseVector ampVec;

  PulseVector errVec;
  PulseVector ampvecpermtest;

  SamplePulseMatrix invcovp;
  PulseMatrix aTaMat; // A-transpose A (matrix)
  PulseVector aTbVec; // A-transpose b (vector)
  PulseVector updateWork; // w (vector)

  SampleDecompLLT covDecomp;
  SampleMatrix covDecompLinv;
  PulseMatrix topleft_work;
  PulseDecompLDLT pulseDecomp;

};

class MahiFit
{
 public:
  MahiFit();
  ~MahiFit() { };

  void setParameters(bool iDynamicPed, double iTS4Thresh, double chiSqSwitch, 
		     bool iApplyTimeSlew, HcalTimeSlew::BiasSetting slewFlavor,
		     double iMeanTime, double iTimeSigmaHPD, double iTimeSigmaSiPM, 
		     const std::vector <int> &iActiveBXs, int iNMaxItersMin, int iNMaxItersNNLS,
		     double iDeltaChiSqThresh, double iNnlsThresh);

  void phase1Apply(const HBHEChannelInfo& channelData, 
		   float& reconstructedEnergy, 
		   float& reconstructedTime, 
		   bool& useTriple,
		   float& chi2);

  void doFit(SampleVector amplitudes, std::vector<float> &correctedOutput, int nbx);

  void setPulseShapeTemplate  (const HcalPulseShapes::Shape& ps);
  void resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps);

  typedef BXVector::Index Index;
  const HcalPulseShapes::Shape* currentPulseShape_=nullptr;

 private:

  void minimize();
  void onePulseMinimize();
  void updateCov();
  void updatePulseShape(double itQ, FullSampleVector &pulseShape, 
			FullSampleVector &pulseDeriv,
			FullSampleMatrix &pulseCov);
  double calculateArrivalTime();
  double calculateChiSq();
  void nnls();
  void resetWorkspace();

  void nnlsUnconstrainParameter(Index idxp);
  void nnlsConstrainParameter(Index minratioidx);

  void eigenSolveSubmatrix(PulseMatrix& mat, PulseVector& invec, PulseVector& outvec, unsigned NP);

  double getSiPMDarkCurrent(double darkCurrent, double fcByPE, double lambda);
  
  mutable nnlsWorkspace nnlsWork_;

  //hard coded in initializer
  const unsigned int fullTSSize_;
  const unsigned int fullTSofInterest_;

  // Python-configurables
  bool dynamicPed_;
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

  //unsigned int bxSize_;
  //int bxOffset_;
  unsigned int bxSizeConf_;
  int bxOffsetConf_;

  //from channelData
  float dt_;
  float darkCurrent_;
  float fcByPe_;

  //holds constant pedestal constraint
  double pedConstraint_;

  unsigned int tsSize_;
  unsigned int tsOffset_;

  unsigned int fullTSOffset_;


  PulseVector ampVecMin_;
  BXVector bxsMin_;
  unsigned int nP_;
  double chiSq_;

  //for pulse shapes
  int cntsetPulseShape_;
  std::unique_ptr<FitterFuncs::PulseShapeFunctor> psfPtr_;
  std::unique_ptr<ROOT::Math::Functor> pfunctor_;
  /*
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
  */
}; 
#endif
