#include <memory>

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentManager.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/HcalTimeSlewRecord.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include <cassert>
#include <vector>
#include <iostream>
#include <iterator>

class HcalTimeSlew;

class HcalPulseContainmentTest : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HcalPulseContainmentTest(const edm::ParameterSet& iConfig);
  ~HcalPulseContainmentTest() override;

private:
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  const edm::ESGetToken<HcalTimeSlew, HcalTimeSlewRecord> tok_slew_;
  const HcalTimeSlew* hcalTimeSlew_delay_;
};

HcalPulseContainmentTest::HcalPulseContainmentTest(const edm::ParameterSet& iConfig)
    : tok_slew_(esConsumes<HcalTimeSlew, HcalTimeSlewRecord>(edm::ESInputTag{"", "HBHE"})),
      hcalTimeSlew_delay_(nullptr) {}

HcalPulseContainmentTest::~HcalPulseContainmentTest() {}

void HcalPulseContainmentTest::beginJob() {}

void HcalPulseContainmentTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  hcalTimeSlew_delay_ = &iSetup.getData(tok_slew_);

  float fixedphase_ns = 6.0;
  float max_fracerror = 0.02;
  std::unique_ptr<HcalPulseContainmentManager> manager;
  manager = std::make_unique<HcalPulseContainmentManager>(max_fracerror);
  manager->setTimeSlew(hcalTimeSlew_delay_);

  HcalDetId hb1(HcalBarrel, 1, 1, 1);
  HcalDetId he1(HcalEndcap, 17, 1, 1);
  double fc = 10.;
  // test re-finding the correction
  double corr1 = manager->correction(hb1, 4, fixedphase_ns, fc);
  double corr2 = manager->correction(hb1, 4, fixedphase_ns, fc);
  assert(corr1 == corr2);
  // fewer toAdd means bigger correction
  double corr3 = manager->correction(hb1, 2, fixedphase_ns, fc);
  assert(corr3 > corr1);
  // HB and HE have the same shape here
  double corr4 = manager->correction(he1, 4, fixedphase_ns, fc);
  assert(corr4 == corr1);
  std::cout << corr1 << " " << corr2 << " " << corr3 << " " << corr4 << " " << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalPulseContainmentTest);
