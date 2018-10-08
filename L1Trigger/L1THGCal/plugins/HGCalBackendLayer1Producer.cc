#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalBackendLayer1ProcessorBase.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

#include <sstream>
#include <memory>


class HGCalBackendLayer1Producer : public edm::stream::EDProducer<> {  
 public:    
  HGCalBackendLayer1Producer(const edm::ParameterSet&);
  ~HGCalBackendLayer1Producer() override { }

  void beginRun(const edm::Run&, 
                        const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  
 private:
  // inputs
  edm::EDGetToken input_cell_, input_sums_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  std::unique_ptr<HGCalBackendLayer1ProcessorBase> backendProcess_;
};

DEFINE_FWK_MODULE(HGCalBackendLayer1Producer);

HGCalBackendLayer1Producer::
HGCalBackendLayer1Producer(const edm::ParameterSet& conf):
input_cell_(consumes<l1t::HGCalTriggerCellBxCollection>(conf.getParameter<edm::InputTag>("InputTriggerCells")))
{
  //setup Backend parameters
  const edm::ParameterSet& beParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& beProcessorName = beParamConfig.getParameter<std::string>("ProcessorName");
  HGCalBackendLayer1ProcessorBase* beProc = HGCalBackendLayer1Factory::get()->create(beProcessorName, beParamConfig);
  backendProcess_.reset(beProc);

  produces<l1t::HGCalClusterBxCollection>(backendProcess_->name());
}

void HGCalBackendLayer1Producer::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es) {
  es.get<CaloGeometryRecord>().get("",triggerGeometry_);
  backendProcess_->setGeometry(triggerGeometry_.product());
}

void HGCalBackendLayer1Producer::produce(edm::Event& e, const edm::EventSetup& es) {

  // Output collections
  std::unique_ptr<l1t::HGCalClusterBxCollection> be_cluster_output( new l1t::HGCalClusterBxCollection );

  // Input collections
  edm::Handle<l1t::HGCalTriggerCellBxCollection> trigCellBxColl;

  e.getByToken(input_cell_, trigCellBxColl);
  backendProcess_->run(trigCellBxColl, *be_cluster_output, es);
  e.put(std::move(be_cluster_output), backendProcess_->name());
}
