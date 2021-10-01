#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/SimpleAlgoGPU.h"

#include <iostream>
#include <functional>
#include <optional>
#include "RecoParticleFlow/PFClusterProducer/plugins/SimplePFGPUAlgos.h"

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
  //edm::EDGetTokenT<RecHitSoAProductType> InputRecHitSoA_Token_;
  //Output Token
  using IProductType = cms::cuda::Product<hcal::RecHitCollection<calo::common::DevStoragePolicy>>;
  const edm::EDGetTokenT<IProductType> InputRecHitSoA_Token_; 
 
  edm::EDPutTokenT<PFRecHitSoAProductType> OutputPFRecHitSoA_Token_;


  hcal::reconstruction::OutputPFRecHitDataGPU PFRecHits_;
  cms::cuda::ContextState cudaState_;

  hcal::RecHitCollection<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>> tmpRecHits;
 
};


PFHBHERechitProducerGPU::PFHBHERechitProducerGPU(edm::ParameterSet const& ps)
    : InputRecHitSoA_Token_{consumes<IProductType>(ps.getParameter<edm::InputTag>("recHitsM0LabelIn"))} {
    //: InputRecHitSoA_Token_{consumes<IProductType>(ps.getParameter<edm::InputTag>("recHitsM0LabelIn"))},
    //OutputPFRecHitSoA_Token_(produces<PFRecHitSoAProductType>(ps.getParameter<std::string>("pfRecHitsLabelCUDAHBHE"))) {
    
    const auto& prodConf = ps.getParameterSetVector("producers")[0];

    const auto& navConf = ps.getParameterSet("navigator");


}


PFHBHERechitProducerGPU::~PFHBHERechitProducerGPU() {}

void PFHBHERechitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
    edm::ParameterSetDescription desc;

    //desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hbheRecHitProducerGPU"});
    desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hltHbherecoGPU"});

    // Prevents the producer and navigator parameter sets from throwing an exception
    // TODO: Replace with a proper parameter set description: twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp
    desc.setAllowAnything();


    cdesc.addWithDefaultLabel(desc);
}


void PFHBHERechitProducerGPU::acquire(edm::Event const& event,
				      edm::EventSetup const& setup,
				      edm::WaitingTaskWithArenaHolder holder) {

  //auto start = std::chrono::high_resolution_clock::now();
  
  auto const& HBHERecHitSoAProduct = event.get(InputRecHitSoA_Token_);
  cms::cuda::ScopedContextAcquire ctx{HBHERecHitSoAProduct, std::move(holder), cudaState_};
  auto const& HBHERecHitSoA = ctx.get(HBHERecHitSoAProduct);
  size_t num_rechits = HBHERecHitSoA.size;
  tmpRecHits.resize(num_rechits);
  //std::cout << "num rechits = " << num_rechits << std::endl;

  if (num_rechits == 0) {
    return;
  }

  // Lambda function to copy arrays to CPU for testing
  auto lambdaToTransfer = [&ctx](auto& dest, auto* src) {
    using vector_type = typename std::remove_reference<decltype(dest)>::type;
    using src_data_type = typename std::remove_pointer<decltype(src)>::type;
    using type = typename vector_type::value_type;
    static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
    cudaCheck(cudaMemcpyAsync(dest.data(), src, dest.size() * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };  

  // Copy rechit raw energy
  lambdaToTransfer(tmpRecHits.energyM0, HBHERecHitSoA.energyM0.get());

  // Copying is done asynchronously, so make sure it's finished before trying to read the CPU values!
  if (cudaStreamQuery(ctx.stream()) != cudaSuccess) cudaCheck(cudaStreamSynchronize(ctx.stream()));

  std::cout<<"tmpRecHits.energyM0.size() = "<<tmpRecHits.energy.size()<<std::endl;
  std::cout<<"First 10 entries:\n";
  for (int i = 0; i < 10; i++) {
    std::cout<<tmpRecHits.energyM0[i]<<std::endl;
  }

  // Allocate PFRecHit device arrays here (temporarily)
  // This should eventually be done once per run for a fixed maximum size
  PFRecHits_.allocate(num_rechits, ctx.stream());
  
  // Entry point for GPU calls 
  hcal::reconstruction::entryPoint_for_PFComputation(HBHERecHitSoA, PFRecHits_, ctx.stream());
}


void PFHBHERechitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};
  ctx.emplace(event, OutputPFRecHitSoA_Token_, std::move(PFRecHits_));

}

DEFINE_FWK_MODULE(PFHBHERechitProducerGPU);
