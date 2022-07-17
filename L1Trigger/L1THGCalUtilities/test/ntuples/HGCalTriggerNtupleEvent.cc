#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"

class HGCalTriggerNtupleEvent : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleEvent(const edm::ParameterSet&);

  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event&, const HGCalTriggerNtupleEventSetup&) final;

private:
  void clear() final;

  int run_;
  int event_;
  int lumi_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleEvent, "HGCalTriggerNtupleEvent");

HGCalTriggerNtupleEvent::HGCalTriggerNtupleEvent(const edm::ParameterSet& conf) : HGCalTriggerNtupleBase(conf) {
  accessEventSetup_ = false;
}

void HGCalTriggerNtupleEvent::initialize(TTree& tree, const edm::ParameterSet&, edm::ConsumesCollector&&) {
  clear();
  tree.Branch("run", &run_, "run/I");
  tree.Branch("event", &event_, "event/I");
  tree.Branch("lumi", &lumi_, "lumi/I");
}

void HGCalTriggerNtupleEvent::fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) {
  run_ = e.id().run();
  lumi_ = e.luminosityBlock();
  event_ = e.id().event();
}

void HGCalTriggerNtupleEvent::clear() {
  run_ = 0;
  lumi_ = 0;
  event_ = 0;
}
