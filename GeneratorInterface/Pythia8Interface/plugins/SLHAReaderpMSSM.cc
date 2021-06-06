#include "GeneratorInterface/Pythia8Interface/interface/SLHAReaderBase.h"

#include "TTree.h"
#include "TString.h"

#include <memory>

class SLHAReaderpMSSM : public SLHAReaderBase {
public:
  SLHAReaderpMSSM(const edm::ParameterSet& conf) : SLHAReaderBase(conf) {}
  ~SLHAReaderpMSSM() override {}

  std::string getSLHA(const std::string& configDesc) override;
};

DEFINE_EDM_PLUGIN(SLHAReaderFactory, SLHAReaderpMSSM, "SLHAReaderpMSSM");

std::string SLHAReaderpMSSM::getSLHA(const std::string& configDesc) {
  const auto& config_fields = splitline(configDesc, '_');
  int chain = std::stoi(config_fields.at(2));
  int iteration = std::stoi(config_fields.at(3));

  auto slhabranch = std::make_unique<TString>();
  auto slhabranch_ptr = slhabranch.get();
  tree_->SetBranchAddress("slhacontent", &slhabranch_ptr);
  tree_->GetEntryWithIndex(chain, iteration);

  return std::string(*slhabranch);
}
