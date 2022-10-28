#ifndef __L1Trigger_L1THGCal_HGCalSortingTruncationConfig_SA_h__
#define __L1Trigger_L1THGCal_HGCalSortingTruncationConfig_SA_h__

namespace l1thgcfirmware {

  class SortingTruncationAlgoConfig {
  public:
    SortingTruncationAlgoConfig(const unsigned maxTCs) : maxTCs_(maxTCs){};

    void setParameters(unsigned maxTCs) { maxTCs_ = maxTCs; };

    void setParameters(const SortingTruncationAlgoConfig& newConfig) { setParameters(newConfig.maxTCs()); }

    unsigned maxTCs() const { return maxTCs_; }

  private:
    unsigned maxTCs_;
  };

}  // namespace l1thgcfirmware

#endif
