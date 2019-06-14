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
  static const unsigned kNDataSize_ = 64;
  static const uint32_t kWaferOffset_ = 3;
  static const uint32_t kWaferMask_ = 0x7;
  static const uint32_t kLinkMask_ = 0x7;

  HGCalTriggerTools triggerTools_;
};

#endif
