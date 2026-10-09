#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeoTesterBase.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class HGCalTriggerGeoTesterManager : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  typedef std::unique_ptr<HGCalTriggerGeoTesterBase> tester_ptr;

public:
  explicit HGCalTriggerGeoTesterManager(const edm::ParameterSet& conf);
  ~HGCalTriggerGeoTesterManager() override = default;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  std::vector<std::pair<tester_ptr, TTree*>> testers_;

  HGCalTriggerGeoTesterEventSetup tester_es_;
  const edm::ESGetToken<HGCalTriggerGeometryBase, CaloGeometryRecord> triggerGeomToken_;
};

DEFINE_FWK_MODULE(HGCalTriggerGeoTesterManager);

HGCalTriggerGeoTesterManager::HGCalTriggerGeoTesterManager(const edm::ParameterSet& conf)
    : triggerGeomToken_(esConsumes<HGCalTriggerGeometryBase, CaloGeometryRecord, edm::Transition::BeginRun>()) {
  usesResource("TFileService");
  edm::Service<TFileService> file_service;
  const std::vector<edm::ParameterSet>& tester_cfgs = conf.getParameterSetVector("Testers");
  for (const auto& tester_cfg : tester_cfgs) {
    const std::string& tester_name = tester_cfg.getParameter<std::string>("TesterName");
    TTree* tree = file_service->make<TTree>(tester_name.c_str(), tester_name.c_str());
    testers_.emplace_back(HGCalTriggerGeoTesterFactory::get()->create(tester_name, tester_cfg), tree);
    testers_.back().first->initialize(tree, tester_cfg);
  }
}

void HGCalTriggerGeoTesterManager::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  tester_es_.geometry = es.getHandle(triggerGeomToken_);
  for (auto& [tester, tree] : testers_) {
    edm::LogPrint("HGCalTriggerGeoTester") << "Running '" << tester->name() << "' tester";
    tester->check(tester_es_);
    tester->fill(tester_es_);
    for (const auto& [error, detids] : tester->errors().errors()) {
      edm::LogError("HGCalTriggerGeoTester")
          << HGCalTriggerGeoTesterErrors::messages.at(error) << " for " << detids.size() << " items"
          << "\n Please check the produced ntuples for more details";
    }
  }
}

void HGCalTriggerGeoTesterManager::analyze(const edm::Event& e, const edm::EventSetup& es) {}
