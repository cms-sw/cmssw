#include "L1Trigger/ME0Trigger/interface/ME0TriggerBuilder.h"

ME0TriggerBuilder::ME0TriggerBuilder(const edm::ParameterSet& conf) {
  config_ = conf;

  for (int endc = 0; endc < 2; endc++) {
    for (int cham = 0; cham < 18; cham++) {
      if ((endc <= 0 || endc > MAX_ENDCAPS) || (cham <= 0 || cham > MAX_CHAMBERS)) {
        edm::LogError("L1ME0TPEmulatorSetupError")
            << "+++ trying to instantiate TMB of illegal ME0 id ["
            << " endcap = " << endc << " chamber = " << cham << "]; skipping it... +++\n";
        continue;
      }
      tmb_[endc][cham].reset(new ME0Motherboard(endc, cham, config_));
    }
  }
}

ME0TriggerBuilder::~ME0TriggerBuilder() {}
