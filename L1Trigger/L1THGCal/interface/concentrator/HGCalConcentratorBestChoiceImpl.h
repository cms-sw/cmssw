#ifndef __L1Trigger_L1THGCal_HGCalConcentratorBestChoiceImpl_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorBestChoiceImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include <vector>

class HGCalConcentratorBestChoiceImpl {
public:
  HGCalConcentratorBestChoiceImpl(const edm::ParameterSet& conf);

  void select(unsigned nLinks,
              unsigned nWafers,
              const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
              std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput);

  void eventSetup(const edm::EventSetup& es) { triggerTools_.eventSetup(es); }

private:
  std::vector<unsigned> nData_;
  static constexpr unsigned kNDataSize_ = 128;
  static constexpr uint32_t kWaferOffset_ = 4;
  static constexpr uint32_t kWaferMask_ = 0x7;
  static constexpr uint32_t kLinkMask_ = 0xF;

  HGCalTriggerTools triggerTools_;
};

#endif
