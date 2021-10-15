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
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

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

  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESHandle<CaloGeometry> geoHandle;
  edm::ESHandle<HcalTopology> topoHandle;

  std::unique_ptr<const HcalTopology> topology_;

  pf::rechit::PersistentDataCPU persistentDataCPU;
  pf::rechit::PersistentDataGPU persistentDataGPU;

  uint32_t nValidDetIds = 0;
  float qTestThresh = 0.;
  std::vector<std::vector<DetId>> neighboursHcal_;
  unsigned int denseIdHcalMax_;
  unsigned int denseIdHcalMin_;
};


PFHBHERechitProducerGPU::PFHBHERechitProducerGPU(edm::ParameterSet const& ps)
    : InputRecHitSoA_Token_{consumes<IProductType>(ps.getParameterSetVector("producers")[0].getParameter<edm::InputTag>("src"))},
      hcalToken_(esConsumes()),
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
  
  auto getIdx = [&](const unsigned int denseid) {
    unsigned index = denseid - denseIdHcalMin_;
    return index;
  };

  auto validNeighbours = [&](const unsigned int denseid) {
    bool ok = true;

    if (denseid < denseIdHcalMin_ || denseid > denseIdHcalMax_) {
      ok = false;
    } else {
      unsigned index = getIdx(denseid);
      if (neighboursHcal_.at(index).size() != 9)
    ok = false;  // the neighbour vector size should be 3x3
    }
    return ok;
  };

    if (!geoHandle.isValid()) {
        topoHandle = setup.getHandle(hcalToken_);
        topology_.release();
        topology_.reset(topoHandle.product());
        
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

        std::vector<unsigned int> vDenseIdHcal;
        neighboursHcal_.clear();
        vDenseIdHcal.reserve(nValidDetIds);
        for (auto hDetId : validBarrelDetIds) {
          vDenseIdHcal.push_back(topology_.get()->detId2denseId(hDetId));
        }
        for (auto hDetId : validEndcapDetIds) {
          vDenseIdHcal.push_back(topology_.get()->detId2denseId(hDetId));
        }
        std::sort(vDenseIdHcal.begin(), vDenseIdHcal.end());

        // Fill a vector of cell neighbours
        denseIdHcalMax_ = *max_element(vDenseIdHcal.begin(), vDenseIdHcal.end());
        denseIdHcalMin_ = *min_element(vDenseIdHcal.begin(), vDenseIdHcal.end());
        neighboursHcal_.resize(denseIdHcalMax_ - denseIdHcalMin_ + 1);

        for (auto denseid : vDenseIdHcal) {
          DetId N(0);
          DetId E(0);
          DetId S(0);
          DetId W(0);
          DetId NW(0);
          DetId NE(0);
          DetId SW(0);
          DetId SE(0);
          std::vector<DetId> neighbours(9, DetId(0));

          // the centre
          unsigned denseid_c = denseid;
          DetId detid_c = topology_.get()->denseId2detId(denseid_c);
          CaloNavigator<HcalDetId> navigator(detid_c, topology_.get());

          HcalDetId hid_c   = HcalDetId(detid_c);

          // Using enum in Geometry/CaloTopology/interface/CaloDirection.h
          // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
          neighbours.at(NONE) = detid_c;

          navigator.home();
          E = navigator.east();
          neighbours.at(EAST) = E;
          if (hid_c.ieta()>0.){ // positive eta: east -> move to smaller |ieta| (finner phi granularity) first
        if (E != DetId(0)) {
          // SE
          SE = navigator.south();
          neighbours.at(SOUTHEAST) = SE;
          // NE
          navigator.home();
          navigator.east();
          NE = navigator.north();
          neighbours.at(NORTHEAST) = NE;
        } 
          } // ieta<0 is handled later.

          navigator.home();
          W = navigator.west();
          neighbours.at(WEST) = W;
          if (hid_c.ieta()<0.){ // negative eta: west -> move to smaller |ieta| (finner phi granularity) first
        if (W != DetId(0)) {
          NW = navigator.north();
          neighbours.at(NORTHWEST) = NW;
          //
          navigator.home();
          navigator.west();
          SW = navigator.south();
          neighbours.at(SOUTHWEST) = SW;
        } 
          } // ieta>0 is handled later.

          navigator.home();
          N = navigator.north();
          neighbours.at(NORTH) = N;
          if (N != DetId(0)) {
            if (hid_c.ieta()<0.) { // negative eta: move in phi first then move to east (coarser phi granularity)
              NE = navigator.east();
              neighbours.at(NORTHEAST) = NE;
            }
            else { // positive eta: move in phi first then move to west (coarser phi granularity)
              NW = navigator.west();
              neighbours.at(NORTHWEST) = NW;
            }
          }

          navigator.home();
          S = navigator.south();
          neighbours.at(SOUTH) = S;
          if (S != DetId(0)) {
            if (hid_c.ieta()>0.){ // positive eta: move in phi first then move to west (coarser phi granularity)
              SW = navigator.west();
              neighbours.at(SOUTHWEST) = SW;
            }
            else { // negative eta: move in phi first then move to east (coarser phi granularity)
              SE = navigator.east();
              neighbours.at(SOUTHEAST) = SE;
            }
          } 

          unsigned index = getIdx(denseid_c);
          neighboursHcal_[index] = neighbours;
        }

        //
        // Check backward compatibility (does a neighbour of a channel have the channel as a neighbour?)
        //
        for (auto denseid : vDenseIdHcal) {
          DetId detid = topology_.get()->denseId2detId(denseid);
          HcalDetId hid   = HcalDetId(detid);
          if (detid==DetId(0)) continue;
          if (!validNeighbours(denseid)) continue;
          std::vector<DetId> neighbours(9, DetId(0));
          unsigned index = getIdx(denseid);
          neighbours = neighboursHcal_.at(index);

          //
          // Loop over neighbours
          int ineighbour=-1;
          for (auto neighbour : neighbours) {
        ineighbour++;
        if (neighbour==DetId(0)) continue;
        //HcalDetId hidn  = HcalDetId(neighbour);
        std::vector<DetId> neighboursOfNeighbour(9, DetId(0));
        std::unordered_set<unsigned int> listOfNeighboursOfNeighbour; // list of neighbours of neighbour
        unsigned denseidNeighbour = topology_.get()->detId2denseId(neighbour);
        if (!validNeighbours(denseidNeighbour)) continue;
        neighboursOfNeighbour = neighboursHcal_.at(getIdx(denseidNeighbour));

        //
        // Loop over neighbours of neighbours
        for (auto neighbourOfNeighbour : neighboursOfNeighbour) {
          if (neighbourOfNeighbour==DetId(0)) continue;
          unsigned denseidNeighbourOfNeighbour = topology_.get()->detId2denseId(neighbourOfNeighbour);    
          if (!validNeighbours(denseidNeighbourOfNeighbour)) continue;
          listOfNeighboursOfNeighbour.insert(denseidNeighbourOfNeighbour);
        }

        //
        if (listOfNeighboursOfNeighbour.find(denseid)==listOfNeighboursOfNeighbour.end()){ 
          // this neighbour is not backward compatible. ignore in the canse of HE phi segmentation change boundary
          if (hid.subdet()==HcalBarrel || hid.subdet()==HcalEndcap) {
            /* std::cout << "This neighbor does not have the original channel as its neighbor. Ignore: "  */
            /*        << detid.det() << " " << hid.ieta() << " " << hid.iphi() << " " << hid.depth() << " "  */
            /*        << neighbour.det() << " " << hidn.ieta() << " " << hidn.iphi() << " " << hidn.depth() */
            /*        << std::endl; */
            neighboursHcal_[index][ineighbour] = DetId(0);
          }
        }
          } // loop over neighbours
        } // loop over vDenseIdHcal
    
        // Test me
        unsigned testDetId = 1158694936;
        std::cout<<"Neighbors of "<<testDetId<<":\n";
        for (auto& n : neighboursHcal_[getIdx(topology_.get()->detId2denseId(testDetId))]) {
            std::cout<<"\t"<<n.rawId()<<std::endl;
        }
        std::cout<<std::endl;

        uint32_t nRHTotal = 0;
        for (const auto& detId : validBarrelDetIds) {
            const auto& pos = hcalBarrelGeo->getGeometry(detId)->getPosition();
            persistentDataCPU.rh_pos[nRHTotal] = make_float3(pos.x(), pos.y(), pos.z());
            persistentDataCPU.rh_detId[nRHTotal] = detId.rawId();
            uint32_t centerIndex = getIdx(topology_.get()->detId2denseId(testDetId));
            for (uint32_t n = 0; n < 8; n++) {
                // cmssdt.cern.ch/lxr/source/RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h#0087
                // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
                // neighboursHcal_[centerIndex][0] is the rechit itself. Skip for neighbour array
                // If no neighbour exists in a direction, the value will be 0
                persistentDataCPU.rh_neighbours[nRHTotal*8 + n] = neighboursHcal_[centerIndex][n+1];
            }
            nRHTotal++;
        }

        
        for (const auto& detId : validEndcapDetIds) {
            const auto& pos = hcalEndcapGeo->getGeometry(detId)->getPosition();
            persistentDataCPU.rh_pos[nRHTotal] = make_float3(pos.x(), pos.y(), pos.z());
            persistentDataCPU.rh_detId[nRHTotal] = detId.rawId(); 
            uint32_t centerIndex = getIdx(topology_.get()->detId2denseId(testDetId));
            for (uint32_t n = 0; n < 8; n++) {
                // neighboursHcal_[centerIndex][0] is the rechit itself. Skip for neighbour array
                persistentDataCPU.rh_neighbours[nRHTotal*8 + n] = neighboursHcal_[centerIndex][n+1];
            }
            nRHTotal++;
        }

        // Copy to GPU
        cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_pos.get(), persistentDataCPU.rh_pos.get(), nRHTotal * sizeof(float3), cudaMemcpyDeviceToHost, ctx.stream()));
        cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_detId.get(), persistentDataCPU.rh_detId.get(), nRHTotal * sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx.stream()));
        cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_neighbours.get(), persistentDataCPU.rh_neighbours.get(), 8 * nRHTotal * sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx.stream()));
    
        // Initialize Cuda constants
        pf::rechit::initializeCudaConstants(validBarrelDetIds.size(), validEndcapDetIds.size(), qTestThresh);
        
        
        //topology_.release();
    }

  if (num_rechits == 0) {
    return;
  }


  // Copy rechit raw energy
  lambdaToTransfer(tmpRecHits.energyM0, HBHERecHitSoA.energyM0.get());
  lambdaToTransfer(tmpRecHits.did, HBHERecHitSoA.did.get());

  // Copying is done asynchronously, so make sure it's finished before trying to read the CPU values!
  if (cudaStreamQuery(ctx.stream()) != cudaSuccess) cudaCheck(cudaStreamSynchronize(ctx.stream()));

  std::cout<<"tmpRecHits.energyM0.size() = "<<tmpRecHits.energy.size()<<std::endl;
  std::cout<<"detId\tenergy\n";
  for (int i = 0; i < 10; i++) {
    std::cout<<tmpRecHits.did[i]<<"\t"<<tmpRecHits.energyM0[i]<<std::endl;
  }

  std::vector<int> sortFailed; 
  for (int i = 0; i < (int)tmpRecHits.did.size()-1; i++) {
    if (tmpRecHits.did[i] > tmpRecHits.did[i+1]) {
        sortFailed.push_back(i);
    }
  }
  if ((int)sortFailed.size() == 0) std::cout<<"Input rechits are sorted!"<<std::endl;
  else {
    std::cout<<"Input rechits are NOT sorted ("<<sortFailed.size()<<" instances)!"<<std::endl;
    for (auto& i: sortFailed) {
        std::cout<<"\ti = "<<i<<"\t"<<tmpRecHits.did[i]<<" -> "<<tmpRecHits.did[i+1]<<std::endl;
    }
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
