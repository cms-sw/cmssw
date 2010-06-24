#ifndef RecoLocalCalo_EcalRecAlgos_ESRecHitFitAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_ESRecHitFitAlgo_HH

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/ESObjects/interface/ESRecHitRatioCuts.h"
#include "CondFormats/ESObjects/interface/ESAngleCorrectionFactors.h"

#include "TF1.h"

class ESRecHitFitAlgo {

 public:

  ESRecHitFitAlgo();
  ~ESRecHitFitAlgo();

  void setESGain(const double& value) { gain_ = value; }
  void setMIPGeV(const double& value) { MIPGeV_ = value; } 
  void setPedestals(const ESPedestals* peds) { peds_ = peds; }
  void setIntercalibConstants(const ESIntercalibConstants* mips) { mips_ = mips; }
  void setChannelStatus(const ESChannelStatus* status) { channelStatus_ = status; }
  void setRatioCuts(const ESRecHitRatioCuts* ratioCuts) { ratioCuts_ = ratioCuts; }
  void setAngleCorrectionFactors(const ESAngleCorrectionFactors* ang) { ang_ = ang; }
  double* EvalAmplitude(const ESDataFrame& digi, double ped) const;
  EcalRecHit reconstruct(const ESDataFrame& digi) const;

 private:

  TF1 *fit_;
  double gain_;
  const ESPedestals *peds_;
  const ESIntercalibConstants *mips_;
  const ESChannelStatus *channelStatus_;
  const ESRecHitRatioCuts *ratioCuts_;
  const ESAngleCorrectionFactors *ang_;
  double MIPGeV_;

};

#endif
