#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalBackendLayer2ProcessorBase.h"

#include <sstream>
#include <memory>


class HGCalBackendLayer2Producer : public edm::stream::EDProducer<> {  
 public:    
  HGCalBackendLayer2Producer(const edm::ParameterSet&);
  ~HGCalBackendLayer2Producer() override { }
  
  void beginRun(const edm::Run&, 
                        const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  // inputs
  edm::EDGetToken input_clusters_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  std::unique_ptr<HGCalBackendLayer2ProcessorBase> backendProcess_;
};

DEFINE_FWK_MODULE(HGCalBackendLayer2Producer);

HGCalBackendLayer2Producer::
HGCalBackendLayer2Producer(const edm::ParameterSet& conf): 
  input_clusters_(consumes<l1t::HGCalClusterBxCollection>(conf.getParameter<edm::InputTag>("InputCluster")))
{
  //setup Backend parameters
  const edm::ParameterSet& beParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& beProcessorName = beParamConfig.getParameter<std::string>("ProcessorName");
  HGCalBackendLayer2ProcessorBase* beProc = HGCalBackendLayer2Factory::get()->create(beProcessorName, beParamConfig);
  backendProcess_.reset(beProc);

  produces<l1t::HGCalMulticlusterBxCollection>(backendProcess_->name());
}

void HGCalBackendLayer2Producer::beginRun(const edm::Run& /*run*/,
                                          const edm::EventSetup& es) 
{                 
  es.get<CaloGeometryRecord>().get("",triggerGeometry_);
  backendProcess_->setGeometry(triggerGeometry_.product());
}

void HGCalBackendLayer2Producer::produce(edm::Event& e, const edm::EventSetup& es) 
{
  // Output collections
  std::unique_ptr<l1t::HGCalMulticlusterBxCollection> be_multicluster_output( new l1t::HGCalMulticlusterBxCollection );

  // Input collections   
  edm::Handle<l1t::HGCalClusterBxCollection> trigCluster2DBxColl;

  e.getByToken(input_clusters_, trigCluster2DBxColl);

  backendProcess_->run(trigCluster2DBxColl, *be_multicluster_output, es);

  e.put(std::move(be_multicluster_output), backendProcess_->name());  
}
