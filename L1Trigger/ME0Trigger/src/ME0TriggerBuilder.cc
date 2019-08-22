#include "L1Trigger/ME0Trigger/interface/ME0TriggerBuilder.h"

ME0TriggerBuilder::ME0TriggerBuilder(const edm::ParameterSet& conf) {
  config_ = conf;

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) {
    for (int cham = 0; cham < MAX_CHAMBERS; cham++) {
      tmb_[endc][cham].reset(new ME0Motherboard(endc, cham, config_));
    }
  }
}

ME0TriggerBuilder::~ME0TriggerBuilder() {}
