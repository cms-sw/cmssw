#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"

class L1TCaloTriggerNtupleBase : public HGCalTriggerNtupleBase {
public:
  L1TCaloTriggerNtupleBase(const edm::ParameterSet& conf)
      : HGCalTriggerNtupleBase(conf),
        branch_name_prefix_(conf.getUntrackedParameter<std::string>("BranchNamePrefix", "")) {}
  ~L1TCaloTriggerNtupleBase() override{};

  std::string branch_name_w_prefix(const std::string name) const { return branch_name_prefix_ + "_" + name; }

private:
  std::string branch_name_prefix_;
};
