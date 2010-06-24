#ifndef RecoLocalCalo_EcalRecAlgos_ESRecHitSimAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_ESRecHitSimAlgo_HH

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/ESObjects/interface/ESRecHitRatioCuts.h"
#include "CondFormats/ESObjects/interface/ESAngleCorrectionFactors.h"

class ESRecHitSimAlgo {

 public:

  ESRecHitSimAlgo();
  ~ESRecHitSimAlgo(){}

  void setESGain(const double& value) { gain_ = value; }
  void setMIPGeV(const double& value) { MIPGeV_ = value; }
  void setPedestals(const ESPedestals* peds) { peds_ = peds; }
  void setIntercalibConstants(const ESIntercalibConstants* mips) { mips_ = mips; }
  void setChannelStatus(const ESChannelStatus* status) { channelStatus_ = status; }
  void setRatioCuts(const ESRecHitRatioCuts* ratioCuts) { ratioCuts_ = ratioCuts; }
  void setAngleCorrectionFactors(const ESAngleCorrectionFactors* ang) { ang_ = ang; }
  void setW0(const double& value) { w0_ = value; } 
  void setW1(const double& value) { w1_ = value; } 
  void setW2(const double& value) { w2_ = value; } 

  double* EvalAmplitude(const ESDataFrame& digi, const double& ped, const double& w0, const double& w1, const double& w2) const;
  EcalRecHit reconstruct(const ESDataFrame& digi) const;

 private:

  int gain_;
  const ESPedestals *peds_;
  const ESIntercalibConstants *mips_;
  const ESChannelStatus *channelStatus_;
  const ESRecHitRatioCuts *ratioCuts_;
  const ESAngleCorrectionFactors *ang_;
  double w0_;
  double w1_;
  double w2_;
  double MIPGeV_;

};

#endif
