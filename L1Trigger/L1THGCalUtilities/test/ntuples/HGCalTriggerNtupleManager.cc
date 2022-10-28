#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class HGCalTriggerNtupleManager : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  typedef std::unique_ptr<HGCalTriggerNtupleBase> ntuple_ptr;

public:
  explicit HGCalTriggerNtupleManager(const edm::ParameterSet& conf);
  ~HGCalTriggerNtupleManager() override = default;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  std::vector<ntuple_ptr> hgc_ntuples_;
  TTree* tree_;

  HGCalTriggerNtupleEventSetup ntuple_es_;
  const edm::ESGetToken<HepPDT::ParticleDataTable, PDTRecord> pdtToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  const edm::ESGetToken<HGCalTriggerGeometryBase, CaloGeometryRecord> triggerGeomToken_;
};

DEFINE_FWK_MODULE(HGCalTriggerNtupleManager);

HGCalTriggerNtupleManager::HGCalTriggerNtupleManager(const edm::ParameterSet& conf)
    : pdtToken_(esConsumes<HepPDT::ParticleDataTable, PDTRecord, edm::Transition::BeginRun>()),
      magfieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      triggerGeomToken_(esConsumes<HGCalTriggerGeometryBase, CaloGeometryRecord, edm::Transition::BeginRun>()) {
  usesResource("TFileService");
  edm::Service<TFileService> file_service;
  tree_ = file_service->make<TTree>("HGCalTriggerNtuple", "HGCalTriggerNtuple");
  const std::vector<edm::ParameterSet>& ntuple_cfgs = conf.getParameterSetVector("Ntuples");
  for (const auto& ntuple_cfg : ntuple_cfgs) {
    const std::string& ntuple_name = ntuple_cfg.getParameter<std::string>("NtupleName");
    hgc_ntuples_.emplace_back(HGCalTriggerNtupleFactory::get()->create(ntuple_name, ntuple_cfg));
    hgc_ntuples_.back()->initialize(*tree_, ntuple_cfg, consumesCollector());
  }
}

void HGCalTriggerNtupleManager::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  ntuple_es_.pdt = es.getHandle(pdtToken_);
  ntuple_es_.magfield = es.getHandle(magfieldToken_);
  ntuple_es_.geometry = es.getHandle(triggerGeomToken_);
}

void HGCalTriggerNtupleManager::analyze(const edm::Event& e, const edm::EventSetup& es) {
  for (auto& hgc_ntuple : hgc_ntuples_) {
    if (hgc_ntuple->accessEventSetup()) {
      hgc_ntuple->fill(e, es);
    } else {
      hgc_ntuple->fill(e, ntuple_es_);
    }
  }

  tree_->Fill();
}
