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
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "RecoLocalCalo/HcalRecProducers/src/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecProducers/src/SimpleAlgoGPU.h"

#include <TFile.h>
#include <TTree.h>

#include <iostream>
#include <functional>
#include <optional>
#include <vector>
#include "RecoParticleFlow/PFClusterProducer/plugins/SimplePFGPUAlgos.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"


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
  using PFRecHitSoAProductType = cms::cuda::Product<pf::rechit::OutputPFRecHitDataGPU>;
  //Input Token
  //edm::EDGetTokenT<RecHitSoAProductType> InputRecHitSoA_Token_;
  //Output Token
  using IProductType = cms::cuda::Product<hcal::RecHitCollection<calo::common::DevStoragePolicy>>;
  const edm::EDGetTokenT<IProductType> InputRecHitSoA_Token_; 
 
  edm::EDPutTokenT<PFRecHitSoAProductType> OutputPFRecHitSoA_Token_;


  pf::rechit::OutputPFRecHitDataGPU PFRecHits_;
  cms::cuda::ContextState cudaState_;

  hcal::RecHitCollection<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>> tmpRecHits;


  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESHandle<CaloGeometry> geoHandle;

  pf::rechit::PersistentDataCPU persistentDataCPU;
  pf::rechit::PersistentDataGPU persistentDataGPU;

  uint32_t nValidDetIds = 0;
  float qTestThresh = 0.;
};


PFHBHERechitProducerGPU::PFHBHERechitProducerGPU(edm::ParameterSet const& ps)
    : InputRecHitSoA_Token_{consumes<IProductType>(ps.getParameterSetVector("producers")[0].getParameter<edm::InputTag>("src"))},
      geomToken_(esConsumes()) {
    //: InputRecHitSoA_Token_{consumes<IProductType>(ps.getParameter<edm::InputTag>("recHitsM0LabelIn"))} {
    
    //: InputRecHitSoA_Token_{consumes<IProductType>(ps.getParameter<edm::InputTag>("recHitsM0LabelIn"))},
    //OutputPFRecHitSoA_Token_(produces<PFRecHitSoAProductType>(ps.getParameter<std::string>("pfRecHitsLabelCUDAHBHE"))) {
    
    const auto& prodConf = ps.getParameterSetVector("producers")[0];
    const std::string& prodName = prodConf.getParameter<std::string>("name");
    std::cout<<"Producer name from config: "<<prodName<<std::endl;
    const auto& qualityConf = prodConf.getParameterSetVector("qualityTests");

    const std::string& qualityTestName = qualityConf[0].getParameter<std::string>("name");
    qTestThresh = (float)qualityConf[0].getParameter<double>("threshold");
    std::cout<<"Quality test name from config: "<<qualityTestName<<std::endl;

    const auto& navConf = ps.getParameterSet("navigator");
}


PFHBHERechitProducerGPU::~PFHBHERechitProducerGPU() {}

void PFHBHERechitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
    edm::ParameterSetDescription desc;

    //desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hbheRecHitProducerGPU"});
    //desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hltHbherecoGPU"});

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
  
  // Lambda function to copy arrays to CPU for testing
  auto lambdaToTransfer = [&ctx](auto& dest, auto* src) {
    using vector_type = typename std::remove_reference<decltype(dest)>::type;
    using src_data_type = typename std::remove_pointer<decltype(src)>::type;
    using type = typename vector_type::value_type;
    static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
    cudaCheck(cudaMemcpyAsync(dest.data(), src, dest.size() * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };  
    
    if (!geoHandle.isValid()) {
        // Get list of valid Det Ids for HCAL barrel & endcap once
        geoHandle = setup.getHandle(geomToken_);
        // get the hcal geometry
        const CaloSubdetectorGeometry* hcalBarrelGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
        const CaloSubdetectorGeometry* hcalEndcapGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

        const std::vector<DetId>& validBarrelDetIds = hcalBarrelGeo->getValidDetIds(DetId::Hcal, HcalBarrel);
        const std::vector<DetId>& validEndcapDetIds = hcalEndcapGeo->getValidDetIds(DetId::Hcal, HcalEndcap);
        nValidDetIds = validBarrelDetIds.size() + validEndcapDetIds.size();
        printf("Valid Det Ids:\n");
        printf("Barrel: %d\nEndcap: %d\n\n", (int)validBarrelDetIds.size(), (int)validEndcapDetIds.size());
        
        // Initialize persistent arrays for rechit positions
        persistentDataCPU.allocate(nValidDetIds, ctx.stream());
        persistentDataGPU.allocate(nValidDetIds, ctx.stream());
        PFRecHits_.allocate(num_rechits, ctx.stream());

        
        uint32_t nRHTotal = 0;
        for (const auto& detId : validBarrelDetIds) {
            const auto& pos = hcalBarrelGeo->getGeometry(detId)->getPosition();
            persistentDataCPU.rh_pos[nRHTotal] = make_float3(pos.x(), pos.y(), pos.z());
            persistentDataCPU.rh_detId[nRHTotal] = detId.rawId(); 
            nRHTotal++;
        }

        
        for (const auto& detId : validEndcapDetIds) {
            const auto& pos = hcalEndcapGeo->getGeometry(detId)->getPosition();
            persistentDataCPU.rh_pos[nRHTotal] = make_float3(pos.x(), pos.y(), pos.z());
            persistentDataCPU.rh_detId[nRHTotal] = detId.rawId(); 
            nRHTotal++;
        }

        // Copy to GPU
        cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_pos.get(), persistentDataCPU.rh_pos.get(), nRHTotal * sizeof(float3), cudaMemcpyDeviceToHost, ctx.stream()));
        cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_detId.get(), persistentDataCPU.rh_detId.get(), nRHTotal * sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx.stream()));
    
        // Initialize Cuda constants
        pf::rechit::initializeCudaConstants(validBarrelDetIds.size(), validEndcapDetIds.size(), qTestThresh);
    }

  if (num_rechits == 0) {
    return;
  }


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
  pf::rechit::entryPoint(HBHERecHitSoA, PFRecHits_, persistentDataGPU, ctx.stream());
}


void PFHBHERechitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};
  ctx.emplace(event, OutputPFRecHitSoA_Token_, std::move(PFRecHits_));

}

DEFINE_FWK_MODULE(PFHBHERechitProducerGPU);
