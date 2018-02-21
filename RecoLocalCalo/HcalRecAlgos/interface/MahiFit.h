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

struct MahiNnlsWorkspace {

  unsigned int nPulseTot;
  unsigned int tsSize;
  unsigned int tsOffset;
  unsigned int fullTSOffset;
  int bxOffset;
  double dt;

  //holds active bunch crossings
  BXVector bxs;  

  //holds data samples
  SampleVector amplitudes;

  //holds inverse covariance matrix
  SampleMatrix invCovMat;

  //holds diagonal noise terms
  SampleVector noiseTerms;

  //holds flat pedestal uncertainty
  SampleMatrix pedConstraint;
  
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
  unsigned int nP;
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
		   float& chi2) const;

  void doFit(std::array<float,3> &correctedOutput, const int nbx) const;

  void setPulseShapeTemplate  (const HcalPulseShapes::Shape& ps,const HcalTimeSlew * hcalTimeSlewDelay);
  void resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps);

  typedef BXVector::Index Index;
  const HcalPulseShapes::Shape* currentPulseShape_=nullptr;
  const HcalTimeSlew* hcalTimeSlewDelay_=nullptr;

 private:

  double minimize() const;
  void onePulseMinimize() const;
  void updateCov() const;
  void updatePulseShape(double itQ, FullSampleVector &pulseShape, 
			FullSampleVector &pulseDeriv,
			FullSampleMatrix &pulseCov) const;

  double calculateArrivalTime() const;
  double calculateChiSq() const;
  void nnls() const;
  void resetWorkspace() const;

  void nnlsUnconstrainParameter(Index idxp) const;
  void nnlsConstrainParameter(Index minratioidx) const;

  void solveSubmatrix(PulseMatrix& mat, PulseVector& invec, PulseVector& outvec, unsigned nP) const;

  inline double getSiPMDarkCurrent(double darkCurrent, double fcByPE, double lambda) const { double mu = darkCurrent * 25 / fcByPE; return sqrt(mu/pow(1-lambda,3)) * fcByPE;}
  
  mutable MahiNnlsWorkspace nnlsWork_;

  //hard coded in initializer
  const unsigned int fullTSSize_;
  const unsigned int fullTSofInterest_;

  static constexpr int pedestalBX_ = 100;

  // Python-configurables
  bool dynamicPed_;
  float ts4Thresh_; 
  float chiSqSwitch_; 

  bool applyTimeSlew_; 
  HcalTimeSlew::BiasSetting slewFlavor_;
  double TSdelay1GeV_=0;

  float meanTime_;
  float timeSigmaHPD_; 
  float timeSigmaSiPM_;

  std::vector <int> activeBXs_;

  int nMaxItersMin_; 
  int nMaxItersNNLS_; 

  float deltaChiSqThresh_; 
  float nnlsThresh_; 

  unsigned int bxSizeConf_;
  int bxOffsetConf_;

  //for pulse shapes
  int cntsetPulseShape_;
  std::unique_ptr<FitterFuncs::PulseShapeFunctor> psfPtr_;
  std::unique_ptr<ROOT::Math::Functor> pfunctor_;

}; 
#endif
