#ifndef __L1Trigger_L1THGCal_HGCalConcentratorCoarsenerImpl_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorCoarsenerImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "L1Trigger/L1THGCal/interface/HGCalCoarseTriggerCellMapping.h"
#include "L1Trigger/L1THGCal/interface/HGCalVFECompressionImpl.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerCellCalibration.h"

class HGCalConcentratorCoarsenerImpl {
public:
  HGCalConcentratorCoarsenerImpl(const edm::ParameterSet& conf);

  void coarsen(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
               std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput);
  void eventSetup(const edm::EventSetup& es) {
    triggerTools_.eventSetup(es);
    coarseTCmapping_.eventSetup(es);
    calibration_.eventSetup(es);
  }

private:
  HGCalTriggerTools triggerTools_;
  bool fixedDataSizePerHGCROC_;
  HGCalCoarseTriggerCellMapping coarseTCmapping_;
  static constexpr int kHighDensityThickness_ = 0;

  HGCalTriggerCellCalibration calibration_;
  HGCalVFECompressionImpl vfeCompression_;

  struct CoarseTC {
    float sumPt;
    float maxMipPt;
    int sumHwPt;
    float sumMipPt;
    unsigned maxId;
  };

  std::unordered_map<uint32_t, CoarseTC> coarseTCs_;

  void updateCoarseTriggerCellMaps(const l1t::HGCalTriggerCell& tc, uint32_t ctcid);
  void assignCoarseTriggerCellEnergy(l1t::HGCalTriggerCell& c, const CoarseTC& ctc) const;
};

#endif
