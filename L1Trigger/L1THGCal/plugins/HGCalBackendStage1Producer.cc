#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalClusteringDummyImpl.h"
#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include <memory>

class HGCalBackendStage1Producer : public edm::stream::EDProducer<> {
public:
  HGCalBackendStage1Producer(const edm::ParameterSet&);
  ~HGCalBackendStage1Producer() override = default;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // inputs
  const edm::EDGetToken input_cell_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  const edm::ESGetToken<HGCalTriggerGeometryBase, CaloGeometryRecord> triggerGeomToken_;

  std::unique_ptr<HGCalBackendStage1ProcessorBase> backendProcess_;

  std::unique_ptr<HGCalClusteringDummyImpl> clusteringDummy_;
};

DEFINE_FWK_MODULE(HGCalBackendStage1Producer);

HGCalBackendStage1Producer::HGCalBackendStage1Producer(const edm::ParameterSet& conf)
    : input_cell_(consumes<l1t::HGCalTriggerCellBxCollection>(conf.getParameter<edm::InputTag>("InputTriggerCells"))),
      triggerGeomToken_(esConsumes<HGCalTriggerGeometryBase, CaloGeometryRecord, edm::Transition::BeginRun>()) {
  //setup Backend parameters
  const edm::ParameterSet& beParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& beProcessorName = beParamConfig.getParameter<std::string>("ProcessorName");

  clusteringDummy_ = std::make_unique<HGCalClusteringDummyImpl>(conf.getParameterSet("C2d_parameters"));

  backendProcess_ = std::unique_ptr<HGCalBackendStage1ProcessorBase>{
      HGCalBackendStage1Factory::get()->create(beProcessorName, beParamConfig)};

  produces<l1t::HGCalClusterBxCollection>(backendProcess_->name());
}

void HGCalBackendStage1Producer::beginRun(const edm::Run& /*run*/, const edm::EventSetup& es) {
  triggerGeometry_ = es.getHandle(triggerGeomToken_);
  backendProcess_->setGeometry(triggerGeometry_.product());
  if (clusteringDummy_)
    clusteringDummy_->setGeometry(triggerGeometry_.product());
}

void HGCalBackendStage1Producer::produce(edm::Event& e, const edm::EventSetup& es) {
  // Output collections
  auto be_cluster_output = std::make_unique<l1t::HGCalClusterBxCollection>();

  // Input collections
  edm::Handle<l1t::HGCalTriggerCellBxCollection> trigCellBxColl;

  // Split trigger cell collection per FPGA
  e.getByToken(input_cell_, trigCellBxColl);

  std::unordered_map<uint32_t, std::vector<edm::Ptr<l1t::HGCalTriggerCell>>> tcs_per_fpga;

  for (unsigned i = 0; i < trigCellBxColl->size(); ++i) {
    edm::Ptr<l1t::HGCalTriggerCell> tc_ptr(trigCellBxColl, i);
    uint32_t module = triggerGeometry_->getModuleFromTriggerCell(tc_ptr->detId());
    uint32_t fpga = triggerGeometry_->getStage1FpgaFromModule(module);
    tcs_per_fpga[fpga].push_back(tc_ptr);
  }

  // Apply truncation per FPGA
  std::vector<edm::Ptr<l1t::HGCalTriggerCell>> truncated_tcs;

  for (auto& fpga_tcs : tcs_per_fpga) {
    backendProcess_->run(fpga_tcs, truncated_tcs);
  }

  // Merge truncated tc collections
  clusteringDummy_->clusterizeDummy(truncated_tcs, *be_cluster_output);

  e.put(std::move(be_cluster_output), backendProcess_->name());
}
