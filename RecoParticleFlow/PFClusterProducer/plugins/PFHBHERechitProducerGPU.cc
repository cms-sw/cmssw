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
#include "FWCore/Framework/interface/ESWatcher.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

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
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  unsigned getIdx(const unsigned);

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

  hcal::PFRecHitCollection<pf::common::VecStoragePolicy<pf::common::CUDAHostAllocatorAlias>> tmpPFRecHits;

  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESHandle<CaloGeometry> geoHandle;
  edm::ESHandle<HcalTopology> topoHandle;

  edm::ESWatcher<HcalRecNumberingRecord> theRecNumberWatcher_; 
  std::unique_ptr<const HcalTopology> topology_;

  pf::rechit::PersistentDataCPU persistentDataCPU;
  pf::rechit::PersistentDataGPU persistentDataGPU;
  pf::rechit::ScratchDataGPU    scratchDataGPU;

  uint32_t nValidBarrelIds = 0, nValidEndcapIds = 0, nValidDetIds = 0;
  float qTestThresh = 0.;
  std::vector<std::vector<DetId>> neighboursHcal_;
  std::vector<unsigned> vDenseIdHcal;
  std::unordered_map<unsigned, unsigned> detIdToIndex; // Mapping of detId to index. Use this index instead of raw detId to encode neighbours
  std::vector<GlobalPoint> validDetIdPositions;
  unsigned denseIdHcalMax_ = 0;
  unsigned denseIdHcalMin_ = 0;

  bool initCuda = true;
  uint32_t nPFRHTotal = 0;
  TTree* tree;
  TFile* f;

  reco::PFRecHitCollection  __pfrechits;
  std::vector<float>  __rh_x;
  std::vector<float>  __rh_y;
  std::vector<float>  __rh_z;
  std::vector<float>  __rh_eta;
  std::vector<float>  __rh_phi;
  std::vector<float> __rh_pt2;
  // rechit neighbours4, neighbours8 vectors
  std::vector<std::vector<int>> __rh_neighbours4;
  std::vector<std::vector<int>> __rh_neighbours8;

//  std::vector<std::vector<int>> __neigh;
};


PFHBHERechitProducerGPU::PFHBHERechitProducerGPU(edm::ParameterSet const& ps)
    : InputRecHitSoA_Token_{consumes<IProductType>(ps.getParameterSetVector("producers")[0].getParameter<edm::InputTag>("src"))},
      hcalToken_(esConsumes<edm::Transition::BeginLuminosityBlock>()),
      geomToken_(esConsumes<edm::Transition::BeginLuminosityBlock>()) {
    //: InputRecHitSoA_Token_{consumes<IProductType>(ps.getParameter<edm::InputTag>("recHitsM0LabelIn"))} {
    
    //: InputRecHitSoA_Token_{consumes<IProductType>(ps.getParameter<edm::InputTag>("recHitsM0LabelIn"))},
    //OutputPFRecHitSoA_Token_(produces<PFRecHitSoAProductType>(ps.getParameter<std::string>("pfRecHitsLabelCUDAHBHE"))) {
    produces<reco::PFRecHitCollection>();
    produces<reco::PFRecHitCollection>("Cleaned");

//    tree = new TTree("tree", "tree");
//    tree->Branch("rechits", "PFRecHitCollection", &__pfrechits);
//    tree->Branch("rechits_x", &__rh_x);
//    tree->Branch("rechits_y", &__rh_y);
//    tree->Branch("rechits_z", &__rh_z);
//    tree->Branch("rechits_eta", &__rh_eta);
//    tree->Branch("rechits_phi", &__rh_phi);
//    tree->Branch("rechits_pt2", &__rh_pt2);
//    tree->Branch("rechits_neighbours4", &__rh_neighbours4);
//    tree->Branch("rechits_neighbours8", &__rh_neighbours8);

    const auto& prodConf = ps.getParameterSetVector("producers")[0];
    const std::string& prodName = prodConf.getParameter<std::string>("name");
    std::cout<<"Producer name from config: "<<prodName<<std::endl;
    const auto& qualityConf = prodConf.getParameterSetVector("qualityTests");

    const std::string& qualityTestName = qualityConf[0].getParameter<std::string>("name");
    qTestThresh = (float)qualityConf[0].getParameter<double>("threshold");
    std::cout<<"Quality test name from config: "<<qualityTestName<<std::endl;

    const auto& navConf = ps.getParameterSet("navigator");
}


PFHBHERechitProducerGPU::~PFHBHERechitProducerGPU() {
    topology_.release();

//  f = new TFile("gpuPFRecHits.root", "recreate");
//  f->cd();
//  tree->Write();
//  f->Close();
//  delete f;
}

void PFHBHERechitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
    edm::ParameterSetDescription desc;

    //desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hbheRecHitProducerGPU"});
    //desc.add<edm::InputTag>("recHitsM0LabelIn", edm::InputTag{"hltHbherecoGPU"});

    // Prevents the producer and navigator parameter sets from throwing an exception
    // TODO: Replace with a proper parameter set description: twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp
    desc.setAllowAnything();


    cdesc.addWithDefaultLabel(desc);
}

unsigned PFHBHERechitProducerGPU::getIdx(const unsigned denseid) {
    return (denseid - denseIdHcalMin_);
}

void PFHBHERechitProducerGPU::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
    if (!theRecNumberWatcher_.check(setup)) return; 
    
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
    nValidBarrelIds = validBarrelDetIds.size();
    nValidEndcapIds = validEndcapDetIds.size();
    nValidDetIds = nValidBarrelIds + nValidEndcapIds; 
        
    neighboursHcal_.clear();
    vDenseIdHcal.clear();
    detIdToIndex.clear();
    validDetIdPositions.clear();

    vDenseIdHcal.reserve(nValidDetIds);
    detIdToIndex.reserve(nValidDetIds);
    validDetIdPositions.reserve(nValidDetIds);

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
    //__neigh.resize(neighboursHcal_.size());
  
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
        
        if (hid_c.subdet() == HcalBarrel)
            validDetIdPositions.emplace_back(hcalBarrelGeo->getGeometry(detid_c)->getPosition());
        else if (hid_c.subdet() == HcalEndcap)
            validDetIdPositions.emplace_back(hcalEndcapGeo->getGeometry(detid_c)->getPosition());
        else
            std::cout<<"Invalid subdetector found for detId "<<hid_c.rawId()<<": "<<hid_c.subdet()<<std::endl;

        detIdToIndex[hid_c.rawId()] = detIdToIndex.size();
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

        
            if (listOfNeighboursOfNeighbour.find(denseid)==listOfNeighboursOfNeighbour.end()){ 
            // this neighbour is not backward compatible. ignore in the canse of HE phi segmentation change boundary
                if (hid.subdet()==HcalBarrel || hid.subdet()==HcalEndcap) {
//                 std::cout << "This neighbor does not have the original channel as its neighbor. Ignore: "  
//                        << detid.det() << " " << hid.ieta() << " " << hid.iphi() << " " << hid.depth() << " "  
//                        << neighbour.det() << " " << hidn.ieta() << " " << hidn.iphi() << " " << hidn.depth() 
//                        << std::endl; 
                    neighboursHcal_[index][ineighbour] = DetId(0);
                }
            }
        } // loop over neighbours
//        std::vector<int> nb(neighboursHcal_[index].size(), -999);
//        for (int i = 0; i < (int)neighboursHcal_[index].size(); i++)
//            nb.at(i) = neighboursHcal_[index].at(i);
//        __neigh[index] = nb;
    } // loop over vDenseIdHcal
    
    
//    TFile* _f = new TFile("gpuNeighbors.root", "recreate");
//    TTree* _t = new TTree("tree", "tree");
//    _t->Branch("neigh", &__neigh);
//    _t->Fill();
//    _f->cd();
//    _t->Write();
//    _f->Close();
    
    initCuda = true;    // (Re)initialize cuda arrays
}

void PFHBHERechitProducerGPU::acquire(edm::Event const& event,
				      edm::EventSetup const& setup,
				      edm::WaitingTaskWithArenaHolder holder) {

    //auto start = std::chrono::high_resolution_clock::now();

    auto const& HBHERecHitSoAProduct = event.get(InputRecHitSoA_Token_);
    //cms::cuda::ScopedContextAcquire ctx{HBHERecHitSoAProduct, std::move(holder), cudaState_};
    cms::cuda::ScopedContextAcquire ctx{HBHERecHitSoAProduct, std::move(holder)};
    auto const& HBHERecHitSoA = ctx.get(HBHERecHitSoAProduct);
    size_t num_rechits = HBHERecHitSoA.size;
    tmpRecHits.resize(num_rechits);
    //std::cout << "num input rechits = " << num_rechits << "\tctx.stream() = " << ctx.stream() << std::endl;

    // Lambda function to copy arrays to CPU for testing
    auto lambdaToTransfer = [&ctx](auto& dest, auto* src) {
        using vector_type = typename std::remove_reference<decltype(dest)>::type;
        using src_data_type = typename std::remove_pointer<decltype(src)>::type;
        using type = typename vector_type::value_type;
        static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
        cudaCheck(cudaMemcpyAsync(dest.data(), src, dest.size() * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
    };  
    
    auto lambdaToTransferSize = [&ctx](auto& dest, auto* src, auto size) {
        using vector_type = typename std::remove_reference<decltype(dest)>::type;
        using src_data_type = typename std::remove_pointer<decltype(src)>::type;
        using type = typename vector_type::value_type;
        static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
        cudaCheck(cudaMemcpyAsync(dest.data(), src, size * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
    };  
  
//    unsigned testDetId = 1158694936;
//    std::cout<<"Neighbors of "<<testDetId<<":\n";
//    for (auto& n : neighboursHcal_[getIdx(topology_.get()->detId2denseId(testDetId))]) {
//        if (n != testDetId) std::cout<<"\t"<<n.rawId()<<std::endl;
//    }
//    std::cout<<std::endl;

    if (initCuda) {
        // Initialize persistent arrays for rechit positions
        persistentDataCPU.allocate(nValidDetIds, ctx.stream());
        persistentDataGPU.allocate(nValidDetIds, ctx.stream());
        scratchDataGPU.allocate(nValidDetIds, ctx.stream());
        //PFRecHits_.allocate(num_rechits, ctx.stream());
        PFRecHits_.allocate(nValidDetIds, ctx.stream());

        uint32_t nRHTotal = 0;
        for (const auto& denseId : vDenseIdHcal) {
            DetId detId = topology_.get()->denseId2detId(denseId);
            HcalDetId hid(detId.rawId());

            persistentDataCPU.rh_pos[nRHTotal] = make_float3(validDetIdPositions.at(nRHTotal).x(), validDetIdPositions.at(nRHTotal).y(), validDetIdPositions.at(nRHTotal).z());
            persistentDataCPU.rh_detId[nRHTotal] = hid.rawId();
            uint32_t centerIndex = getIdx(denseId);

            for (uint32_t n = 0; n < 8; n++) {
                // cmssdt.cern.ch/lxr/source/RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h#0087
                // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
                // neighboursHcal_[centerIndex][0] is the rechit itself. Skip for neighbour array
                // If no neighbour exists in a direction, the value will be 0
                if (neighboursHcal_[centerIndex][n+1].rawId() != 0) {
                    persistentDataCPU.rh_neighbours[nRHTotal*8 + n] = detIdToIndex[neighboursHcal_[centerIndex][n+1].rawId()];
                }
                else
                    persistentDataCPU.rh_neighbours[nRHTotal*8 + n] = -1;
            }
            nRHTotal++;
        }
        
        // Copy to GPU
        cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_pos.get(), persistentDataCPU.rh_pos.get(), nRHTotal * sizeof(float3), cudaMemcpyHostToDevice, ctx.stream()));
        cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_detId.get(), persistentDataCPU.rh_detId.get(), nRHTotal * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx.stream()));
        cudaCheck(cudaMemcpyAsync(persistentDataGPU.rh_neighbours.get(), persistentDataCPU.rh_neighbours.get(), 8 * nRHTotal * sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));

        // Initialize Cuda constants
        pf::rechit::initializeCudaConstants(nValidBarrelIds, nValidEndcapIds, qTestThresh);

        initCuda = false;
    }

  if (num_rechits == 0) {
    return;
  }


  // Copy rechit raw energy
  lambdaToTransfer(tmpRecHits.timeM0, HBHERecHitSoA.timeM0.get());
  lambdaToTransfer(tmpRecHits.energyM0, HBHERecHitSoA.energyM0.get());
  lambdaToTransfer(tmpRecHits.energy, HBHERecHitSoA.energy.get());
  lambdaToTransfer(tmpRecHits.chi2, HBHERecHitSoA.chi2.get());
  lambdaToTransfer(tmpRecHits.did, HBHERecHitSoA.did.get());

  // Copying is done asynchronously, so make sure it's finished before trying to read the CPU values!
  if (cudaStreamQuery(ctx.stream()) != cudaSuccess) cudaCheck(cudaStreamSynchronize(ctx.stream()));


//  std::vector<int> sortFailed; 
//  for (int i = 0; i < (int)tmpRecHits.did.size()-1; i++) {
//    if (tmpRecHits.did[i] > tmpRecHits.did[i+1]) {
//        sortFailed.push_back(i);
//    }
//  }
//  if ((int)sortFailed.size() == 0) std::cout<<"Input rechits are sorted!"<<std::endl;
//  else {
//    std::cout<<"Input rechits are NOT sorted ("<<sortFailed.size()<<" instances)!"<<std::endl;
//    for (auto& i: sortFailed) {
//        std::cout<<"\ti = "<<i<<"\t"<<tmpRecHits.did[i]<<" -> "<<tmpRecHits.did[i+1]<<std::endl;
//    }
//  }
  
  // Entry point for GPU calls 
  pf::rechit::entryPoint(HBHERecHitSoA, PFRecHits_, persistentDataGPU, scratchDataGPU, ctx.stream());

  if (cudaStreamQuery(ctx.stream()) != cudaSuccess) cudaCheck(cudaStreamSynchronize(ctx.stream()));
  // For testing, copy back PFRecHit SoA data to CPU
  //cudaDeviceSynchronize();
  nPFRHTotal = PFRecHits_.PFRecHits.size + PFRecHits_.PFRecHits.sizeCleaned;
  tmpPFRecHits.resize(nPFRHTotal);
  tmpPFRecHits.size = PFRecHits_.PFRecHits.size;
  tmpPFRecHits.sizeCleaned = PFRecHits_.PFRecHits.sizeCleaned;
  
  
  lambdaToTransferSize(tmpPFRecHits.pfrh_depth, PFRecHits_.PFRecHits.pfrh_depth.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_layer, PFRecHits_.PFRecHits.pfrh_layer.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_detId, PFRecHits_.PFRecHits.pfrh_detId.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_neighbours, PFRecHits_.PFRecHits.pfrh_neighbours.get(), 8*nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_neighbourInfos, PFRecHits_.PFRecHits.pfrh_neighbourInfos.get(), 8*nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_time, PFRecHits_.PFRecHits.pfrh_time.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_energy, PFRecHits_.PFRecHits.pfrh_energy.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_pt2, PFRecHits_.PFRecHits.pfrh_pt2.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_x, PFRecHits_.PFRecHits.pfrh_x.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_y, PFRecHits_.PFRecHits.pfrh_y.get(), nPFRHTotal);
  lambdaToTransferSize(tmpPFRecHits.pfrh_z, PFRecHits_.PFRecHits.pfrh_z.get(), nPFRHTotal);
  if (cudaStreamQuery(ctx.stream()) != cudaSuccess) cudaCheck(cudaStreamSynchronize(ctx.stream()));
}


void PFHBHERechitProducerGPU::produce(edm::Event& event, edm::EventSetup const& setup) {
  //cms::cuda::ScopedContextProduce ctx{cudaState_};
  //ctx.emplace(event, OutputPFRecHitSoA_Token_, std::move(PFRecHits_));
 
 
  const CaloSubdetectorGeometry* hcalBarrelGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const CaloSubdetectorGeometry* hcalEndcapGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalEndcap);
  auto pfrhLegacy = std::make_unique<reco::PFRecHitCollection>();
  auto pfrhLegacyCleaned = std::make_unique<reco::PFRecHitCollection>();
  
  pfrhLegacy->reserve(tmpPFRecHits.size);
  pfrhLegacyCleaned->reserve(tmpPFRecHits.sizeCleaned);
  for (unsigned i = 0; i < nPFRHTotal; i++) {
    HcalDetId hid(tmpPFRecHits.pfrh_detId[i]);
    
    std::shared_ptr<const CaloCellGeometry> thisCell = nullptr;
    PFLayer::Layer layer = PFLayer::HCAL_BARREL1;
    switch (hid.subdet()) {
      case HcalBarrel:
        thisCell = hcalBarrelGeo->getGeometry(hid);
        layer = PFLayer::HCAL_BARREL1;
        break;

      case HcalEndcap:
        thisCell = hcalEndcapGeo->getGeometry(hid);
        layer = PFLayer::HCAL_ENDCAP;
        break;
      default:
        break;
    }
    reco::PFRecHit pfrh(thisCell, hid.rawId(), layer, tmpPFRecHits.pfrh_energy[i]);
    pfrh.setTime(tmpPFRecHits.pfrh_time[i]);
    pfrh.setDepth(hid.depth());
    //pfrh.setDepth(tmpPFRecHits.pfrh_depth[i]);

    
    std::vector<int> etas     = {0,  1,  0, -1,  1,  1, -1, -1};
    std::vector<int> phis     = {1,  1, -1, -1,  0, -1,  0,  1};
    std::vector<int> gpuOrder = {0,  4,  1,  5,  2,  6,  3,  7};
    for (int n = 0; n < 8; n++) {
        int neighId = tmpPFRecHits.pfrh_neighbours[i*8+gpuOrder[n]];
        if (i < tmpPFRecHits.size && neighId > -1 && neighId < (int)tmpPFRecHits.size)
            pfrh.addNeighbour(etas[n], phis[n], 0, neighId); 
    }

    if (i < tmpPFRecHits.size)
        pfrhLegacy->push_back(pfrh);
    else
        pfrhLegacyCleaned->push_back(pfrh);
  }
 
  printf("pfrhLegacy->size() = %d\tpfrhLegacyCleaned->size() = %d\n\n", (int)pfrhLegacy->size(), (int)pfrhLegacyCleaned->size());


  __pfrechits = *pfrhLegacy;
  __rh_x.clear();
  __rh_x.reserve((int)pfrhLegacy->size());
  __rh_y.clear();
  __rh_y.reserve((int)pfrhLegacy->size());
  __rh_z.clear();
  __rh_z.reserve((int)pfrhLegacy->size());
  __rh_eta.clear();
  __rh_eta.reserve((int)pfrhLegacy->size());
  __rh_phi.clear();
  __rh_phi.reserve((int)pfrhLegacy->size());
  __rh_pt2.clear();
  __rh_pt2.reserve((int)pfrhLegacy->size());
  __rh_neighbours4.clear();
  __rh_neighbours4.reserve((int)pfrhLegacy->size());
  __rh_neighbours8.clear();
  __rh_neighbours8.reserve((int)pfrhLegacy->size());
  
  for (auto& pfrh : *pfrhLegacy) {
    auto pos = pfrh.position();
    __rh_x.push_back(pos.x());
    __rh_y.push_back(pos.y());
    __rh_z.push_back(pos.z());
    __rh_eta.push_back(pfrh.positionREP().eta());
    __rh_phi.push_back(pfrh.positionREP().phi());
    __rh_pt2.push_back(pfrh.pt2());
    std::vector<int> n4, n8;
    for (auto n : pfrh.neighbours4()) {
        n4.push_back(n);
    }
  
    for (auto n : pfrh.neighbours8()) {
        n8.push_back(n);
    }

    __rh_neighbours4.push_back(n4);
    __rh_neighbours8.push_back(n8);
  }
//  tree->Fill();
//  tree = new TTree("tree", "tree");
//  tree->Branch("detId", &tmpRecHits.did);
//  tree->Branch("chi2", &tmpRecHits.chi2);
//  tree->Branch("energy", &tmpRecHits.energy);
//  tree->Branch("energyM0", &tmpRecHits.energyM0);
//  tree->Branch("timeM0", &tmpRecHits.timeM0);
//  tree->Branch("pfrh_depth", &tmpPFRecHits.pfrh_depth);
//  tree->Branch("pfrh_energy", &tmpPFRecHits.pfrh_energy);
//  tree->Fill();
//
//  f = new TFile("inputRechits.root", "recreate");
//  f->cd();
//  tree->Write();
//  f->Close();
//  delete f;
  
  event.put(std::move(pfrhLegacy), "");
  event.put(std::move(pfrhLegacyCleaned), "Cleaned");

  tmpPFRecHits.resize(0);
}

DEFINE_FWK_MODULE(PFHBHERechitProducerGPU);
