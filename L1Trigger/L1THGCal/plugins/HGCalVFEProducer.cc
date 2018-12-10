#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include <memory>


class HGCalVFEProducer : public edm::stream::EDProducer<>  {  
 public:    
  HGCalVFEProducer(const edm::ParameterSet&);
  ~HGCalVFEProducer() override { }
  
  void beginRun(const edm::Run&, 
                        const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  
  private:
  // inputs
  edm::EDGetToken inputee_, inputfh_, inputbh_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  
  std::unique_ptr<HGCalVFEProcessorBase> vfeProcess_;

};

DEFINE_FWK_MODULE(HGCalVFEProducer);

HGCalVFEProducer::
HGCalVFEProducer(const edm::ParameterSet& conf):
  inputee_(consumes<HGCalDigiCollection>(conf.getParameter<edm::InputTag>("eeDigis"))),
  inputfh_(consumes<HGCalDigiCollection>(conf.getParameter<edm::InputTag>("fhDigis"))),
  inputbh_(consumes<HGCalDigiCollection>(conf.getParameter<edm::InputTag>("bhDigis")))
{   
  //setup VFE parameters
  const edm::ParameterSet& vfeParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& vfeProcessorName = vfeParamConfig.getParameter<std::string>("ProcessorName");
  HGCalVFEProcessorBase* vfeProc = HGCalVFEProcessorBaseFactory::get()->create(vfeProcessorName, vfeParamConfig);
  vfeProcess_.reset(vfeProc);
  
  produces<l1t::HGCalTriggerCellBxCollection>(vfeProcess_->name());
  produces<l1t::HGCalTriggerSumsBxCollection>(vfeProcess_->name());

}

void HGCalVFEProducer::beginRun(const edm::Run& /*run*/, 
                                const edm::EventSetup& es) {
  es.get<CaloGeometryRecord>().get(triggerGeometry_);
  vfeProcess_->setGeometry(triggerGeometry_.product());
}

void HGCalVFEProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  
  // Output collections
  auto vfe_trigcell_output = std::make_unique<l1t::HGCalTriggerCellBxCollection>();
  auto vfe_trigsums_output = std::make_unique<l1t::HGCalTriggerSumsBxCollection>();

  // Input collections
  edm::Handle<HGCalDigiCollection> ee_digis_h;
  edm::Handle<HGCalDigiCollection> fh_digis_h;
  edm::Handle<HGCalDigiCollection> bh_digis_h;

  e.getByToken(inputee_,ee_digis_h);
  e.getByToken(inputfh_,fh_digis_h);
  e.getByToken(inputbh_,bh_digis_h);

  const HGCalDigiCollection& ee_digis = *ee_digis_h;
  const HGCalDigiCollection& fh_digis = *fh_digis_h;
  const HGCalDigiCollection& bh_digis = *bh_digis_h;
  
  // Processing DigiCollections and putting the results into the HGCalTriggerCellBxCollection
  vfeProcess_->run(ee_digis, *vfe_trigcell_output, es);         
  vfeProcess_->run(fh_digis, *vfe_trigcell_output, es);   
  vfeProcess_->run(bh_digis, *vfe_trigcell_output, es);   

  // Put in the event  
  e.put(std::move(vfe_trigcell_output), vfeProcess_->name());
  // At the moment the HGCalTriggerSumsBxCollection is empty 
  e.put(std::move(vfe_trigsums_output), vfeProcess_->name());

}
