#include "L1Trigger/L1TGEM/interface/ME0TriggerBuilder.h"

ME0TriggerBuilder::ME0TriggerBuilder(const edm::ParameterSet& conf) {
  config_ = conf;

  for (int endc = 0; endc < MAX_ENDCAPS; endc++) {
    for (int cham = 0; cham < MAX_CHAMBERS; cham++) {
      tmb_[endc][cham].reset(new ME0Motherboard(endc, cham, config_));
    }
  }
}

ME0TriggerBuilder::~ME0TriggerBuilder() {}

void ME0TriggerBuilder::build(const ME0PadDigiCollection* me0Pads, ME0TriggerDigiCollection& oc_trig) {
  for (int endc = 0; endc < 2; endc++) {
    for (int cham = ME0DetId::minChamberId; cham < ME0DetId::maxChamberId; cham++) {
      ME0Motherboard* tmb = tmb_[endc][cham].get();
      tmb->setME0Geometry(me0_g);

      // 0th layer means whole chamber.
      const int region(endc == 0 ? -1 : 1);
      ME0DetId detid(region, 0, cham + 1, 0);

      // Run processors only if chamber exists in geometry.
      if (tmb == nullptr || me0_g->chamber(detid) == nullptr)
        continue;

      tmb->run(me0Pads);

      const std::vector<ME0TriggerDigi>& trigV = tmb->readoutTriggers();

      if (!trigV.empty()) {
        LogTrace("L1ME0Trigger") << "ME0TriggerBuilder got results in " << detid << std::endl
                                 << "Put " << trigV.size() << " Trigger digi" << ((trigV.size() > 1) ? "s " : " ")
                                 << "in collection\n";
        oc_trig.put(std::make_pair(trigV.begin(), trigV.end()), detid);
      }
    }
  }
}
