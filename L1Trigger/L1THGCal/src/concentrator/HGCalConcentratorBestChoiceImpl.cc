#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorBestChoiceImpl.h"

constexpr unsigned HGCalConcentratorBestChoiceImpl::kNDataSize_;
constexpr uint32_t HGCalConcentratorBestChoiceImpl::kWaferOffset_;
constexpr uint32_t HGCalConcentratorBestChoiceImpl::kWaferMask_;
constexpr uint32_t HGCalConcentratorBestChoiceImpl::kLinkMask_;

HGCalConcentratorBestChoiceImpl::HGCalConcentratorBestChoiceImpl(const edm::ParameterSet& conf)
    : nData_(conf.getParameter<std::vector<unsigned>>("NData")) {
  if (nData_.size() != kNDataSize_) {
    throw cms::Exception("BadInitialization") << "NData vector must be of size " << kNDataSize_;
  }
}

void HGCalConcentratorBestChoiceImpl::select(unsigned nLinks,
                                             unsigned nWafers,
                                             const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
                                             std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput) {
  trigCellVecOutput = trigCellVecInput;
  // sort, reverse order
  std::sort(
      trigCellVecOutput.begin(),
      trigCellVecOutput.end(),
      [](const l1t::HGCalTriggerCell& a, const l1t::HGCalTriggerCell& b) -> bool { return a.mipPt() > b.mipPt(); });

  uint32_t nLinksIndex = 0;
  if (nLinks > kLinkMask_) {
    throw cms::Exception("BadConfig") << "BestChoice: Nlinks=" << nLinks
                                      << " larger then the max supported number of links " << kLinkMask_;
  }
  nLinksIndex |= ((nLinks - 1) & kLinkMask_);
  nLinksIndex |= (((nWafers - 1) & kWaferMask_) << kWaferOffset_);
  unsigned nData = nData_.at(nLinksIndex);
  if (nData == 0) {
    throw cms::Exception("BadConfig") << "BestChoice: NData=0 for "
                                      << " NWafers=" << nWafers << " and NLinks=" << nLinks;
  }
  // keep only N trigger cells
  if (trigCellVecOutput.size() > nData)
    trigCellVecOutput.resize(nData);
}
