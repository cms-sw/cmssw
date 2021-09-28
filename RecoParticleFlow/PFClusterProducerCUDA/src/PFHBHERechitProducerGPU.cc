#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/SimpleAlgoGPU.h"


#include <functional>
#include <optional>
#include "RecoParticleFlow/PFClusterProducerCUDA/src/SimplePFGPUAlgos.h"

class PFHBHERechitProducerGPU : public edm::stream::EDProducer <edm::ExternalWork> {
public:
  explicit PFHBHERechitProducerGPU(edm::ParameterSet const&);
  ~PFHBHERechitProducerGPU() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  
private:
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, edm::EventSetup const&) override;


  // Input Product Type
  using RecHitSoAProductType = cms::cuda::Product<hcal::reconstruction::OutputDataGPU>;
  //Output Product Type
  using PFRecHitSoAProductType = cms::cuda::Product<hcal::reconstruction::OutputPFRecHitDataGPU>;
  //Input Token
  edm::EDGetTokenT<RecHitSoAProductType> InputRecHitSoA_Token_;
  //Output Token
  edm::EDPutTokenT<PFRecHitSoAProductType> OutputPFRecHitSoA_Token_;

  
  hcal::reconstruction::OutputPFRecHitDataGPU PFRecHits_;
  //hcal::reconstruction::
  cms::cuda::ContextState cudaState_;
  
};


PFHBHERechitProducerGPU::PFHBHERechitProducerGPU(edm::ParameterSet const& ps)
  : InputRecHitSoA_Token_(consumes<RecHitSoAProductType>(ps.getParameter<edm::InputTag>("recHitsLabelCUDAHBHE"))),
    OutputPFRecHitSoA_Token_(produces<PFRecHitSoAProductType>(ps.getParameter<std::string>("pfRecHitsLabelCUDAHBHE"))) {

    
}


PFHBHERechitProducerGPU::~PFHBHERechitProducerGPU() {}

void PFHBHERechitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {}


void PFHBHERechitProducerGPU::acquire(edm::Event const& event,
				      edm::EventSetup const& setup,
				      edm::WaitingTaskWithArenaHolder holder) {

  //auto start = std::chrono::high_resolution_clock::now();
  

  auto const& HBHERecHitSoAProduct = event.get(InputRecHitSoA_Token_);
  cms::cuda::ScopedContextAcquire ctx{HBHERecHitSoAProduct, std::move(holder), cudaState_};
  auto const& HBHERecHitSoA = ctx.get(HBHERecHitSoAProduct);
  //size_t RecHits_size = sizeof(*(HBHERecHitSoA)->rechits.energy.get());
  auto const recHits_en = HBHERecHitSoA.recHits.energy.get();
  size_t num_rechits = sizeof(recHits_en);
  PFRecHits_.allocate(num_rechits, ctx.stream());
  std::cout<<typeid(recHits_en).name()<<std::endl;
  //auto const& HBHERecHits_asInput = hcal::reconstruction::OutputDataGPU(HBHERecHitSoA);
  hcal::reconstruction::entryPoint_for_PFComputation(HBHERecHitSoA, PFRecHits_, ctx.stream());

}


void PFHBHERechitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};
  ctx.emplace(event, OutputPFRecHitSoA_Token_, std::move(PFRecHits_));

}

DEFINE_FWK_MODULE(PFHBHERechitProducerGPU);
//DEFINE_FWK_PSET_DESC_FILLER(PFHBHERechitProducerGPU);
