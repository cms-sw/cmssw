#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalBackendLayer2ProcessorBase.h"

#include <sstream>
#include <memory>


class HGCalBackendLayer2Producer : public edm::stream::EDProducer<> {  
 public:    
  HGCalBackendLayer2Producer(const edm::ParameterSet&);
  ~HGCalBackendLayer2Producer() { }
  
  virtual void beginRun(const edm::Run&, 
                        const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  // inputs
  edm::EDGetToken input_cluster2D;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  std::unique_ptr<HGCalBackendLayer2ProcessorBase> backendProcess_;
};

DEFINE_FWK_MODULE(HGCalBackendLayer2Producer);

HGCalBackendLayer2Producer::
HGCalBackendLayer2Producer(const edm::ParameterSet& conf): 
  input_cluster2D(consumes<l1t::HGCalClusterBxCollection>(conf.getParameter<edm::InputTag>("cluster2DCollection_be")))
{   
  //setup Backend parameters
  const edm::ParameterSet& beParamConfig = conf.getParameterSet("Backendparam");
  const std::string& beProcessorName = beParamConfig.getParameter<std::string>("BeProcessorLayer2Name");
  HGCalBackendLayer2ProcessorBase* beProc = HGCalBackendLayer2Factory::get()->create(beProcessorName, beParamConfig);
  backendProcess_.reset(beProc);

  backendProcess_->setProduces3D(*this);
}

void HGCalBackendLayer2Producer::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es) 
{				  
  es.get<CaloGeometryRecord>().get("",triggerGeometry_);
  backendProcess_->setGeometry(triggerGeometry_.product());
}

void HGCalBackendLayer2Producer::produce(edm::Event& e, const edm::EventSetup& es) 
{    
  edm::Handle<l1t::HGCalClusterBxCollection> trigCluster2DBxColl;
  
  e.getByToken(input_cluster2D, trigCluster2DBxColl);

  //const l1t::HGCalClusterBxCollection& trigCluster2D = *trigCluster2DBxColl;
    
  backendProcess_->reset3D();    
  backendProcess_->run3D(trigCluster2DBxColl,es,e);
  //backendProcess_->putInEvent3D(e);  
}
