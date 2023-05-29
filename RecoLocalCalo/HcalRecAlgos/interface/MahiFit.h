#ifndef RecoLocalCalo_HcalRecAlgos_MahiFit_HH
#define RecoLocalCalo_HcalRecAlgos_MahiFit_HH

#include <climits>
#include <utility>
#include <memory>

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/EigenMatrixTypes.h"
#include "DataFormats/HcalRecHit/interface/HBHEChannelInfo.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFunctor.h"

struct MahiNnlsWorkspace {
  unsigned int nPulseTot;
  unsigned int tsSize = 0U;
  unsigned int tsOffset;
  int bxOffset;
  int maxoffset;
  float dt;

  //holds active bunch crossings
  BXVector bxs;

  //holds data samples
  SampleVector amplitudes;

  //holds diagonal noise terms
  SampleVector noiseTerms;

  //holds diagonal pedestal noise terms
  SampleVector pedVals;

  //holds flat pedestal uncertainty
  float pedVal;

  float noisecorr;

  //holds full covariance matrix for a pulse shape
  //varied in time
  std::array<SampleMatrix, MaxPVSize> pulseCovArray;

  //holds matrix of pulse shape templates for each BX
  SamplePulseMatrix pulseMat;

  //holds matrix of pulse shape derivatives for each BX
  SamplePulseMatrix pulseDerivMat;

  //for FNNLS algorithm
  unsigned int nP;
  PulseVector ampVec;

  SamplePulseMatrix invcovp;
  PulseMatrix aTaMat;  // A-transpose A (matrix)
  PulseVector aTbVec;  // A-transpose b (vector)

  SampleDecompLLT covDecomp;
};

struct MahiDebugInfo {
  int nSamples;
  int soi;

  bool use3;

  float inTimeConst;
  float inDarkCurrent;
  float inPedAvg;
  float inGain;

  float inNoiseADC[MaxSVSize];
  float inNoiseDC[MaxSVSize];
  float inNoisePhoto[MaxSVSize];
  float inPedestal[MaxSVSize];

  float totalUCNoise[MaxSVSize];

  float mahiEnergy;
  float chiSq;
  float arrivalTime;
  float pedEnergy;
  float ootEnergy[7];

  float count[MaxSVSize];
  float inputTS[MaxSVSize];
  int inputTDC[MaxSVSize];
  float itPulse[MaxSVSize];
  float ootPulse[7][MaxSVSize];
};

class MahiFit {
public:
  MahiFit();
  ~MahiFit(){};

  void setParameters(bool iDynamicPed,
                     double iTS4Thresh,
                     double chiSqSwitch,
                     bool iApplyTimeSlew,
                     HcalTimeSlew::BiasSetting slewFlavor,
                     bool iCalculateArrivalTime,
                     int iTimeAlgo,
                     double iThEnergeticPulses,
                     double iMeanTime,
                     double iTimeSigmaHPD,
                     double iTimeSigmaSiPM,
                     const std::vector<int>& iActiveBXs,
                     int iNMaxItersMin,
                     int iNMaxItersNNLS,
                     double iDeltaChiSqThresh,
                     double iNnlsThresh);

  void phase1Apply(const HBHEChannelInfo& channelData,
                   float& reconstructedEnergy,
                   float& soiPlusOneEnergy,
                   float& reconstructedTime,
                   bool& useTriple,
                   float& chi2) const;

  void phase1Debug(const HBHEChannelInfo& channelData, MahiDebugInfo& mdi) const;

  void doFit(std::array<float, 4>& correctedOutput, const int nbx) const;

  void setPulseShapeTemplate(int pulseShapeId,
                             const HcalPulseShapes& ps,
                             bool hasTimeInfo,
                             const HcalTimeSlew* hcalTimeSlewDelay,
                             unsigned int nSamples,
                             const float gain);

  typedef BXVector::Index Index;
  const HcalTimeSlew* hcalTimeSlewDelay_ = nullptr;

  float thEnergeticPulses_;
  float thEnergeticPulsesFC_;

private:
  typedef std::pair<int, std::shared_ptr<FitterFuncs::PulseShapeFunctor> > ShapeWithId;

  const float minimize() const;
  void onePulseMinimize() const;
  void updateCov(const SampleMatrix& invCovMat) const;
  void resetPulseShapeTemplate(int pulseShapeId, const HcalPulseShapes& ps, unsigned int nSamples);

  float ccTime(const float itQ) const;
  void updatePulseShape(const float itQ,
                        FullSampleVector& pulseShape,
                        FullSampleVector& pulseDeriv,
                        FullSampleMatrix& pulseCov) const;

  float calculateArrivalTime(const unsigned int iBX) const;
  float calculateChiSq() const;
  void nnls() const;
  void resetWorkspace() const;

  void nnlsUnconstrainParameter(Index idxp) const;
  void nnlsConstrainParameter(Index minratioidx) const;

  void solveSubmatrix(PulseMatrix& mat, PulseVector& invec, PulseVector& outvec, unsigned nP) const;

  mutable MahiNnlsWorkspace nnlsWork_;

  //hard coded in initializer
  static constexpr int pedestalBX_ = 100;

  // used to restrict returned time value to a 25 ns window centered
  // on the nominal arrival time
  static constexpr float timeLimit_ = 12.5f;

  // Python-configurables
  int timeAlgo_;

  bool dynamicPed_;
  float ts4Thresh_;
  float chiSqSwitch_;

  bool applyTimeSlew_;
  HcalTimeSlew::BiasSetting slewFlavor_;
  float tsDelay1GeV_ = 0.f;
  float norm_ = (1.f / std::sqrt(12));

  bool calculateArrivalTime_;
  float meanTime_;
  float timeSigmaHPD_;
  float timeSigmaSiPM_;

  std::vector<int> activeBXs_;

  int nMaxItersMin_;
  int nMaxItersNNLS_;

  float deltaChiSqThresh_;
  float nnlsThresh_;

  unsigned int bxSizeConf_;
  int bxOffsetConf_;

  //for pulse shapes
  int currentPulseShapeId_ = INT_MIN;
  int cntsetPulseShape_ = 0;
  FitterFuncs::PulseShapeFunctor* psfPtr_ = nullptr;
  std::vector<ShapeWithId> knownPulseShapes_;
};

#endif
