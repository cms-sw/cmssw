#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

#include "CudaPFCommon.h"
#include "DeclsForKernels.h"
#include "PFClusterCudaHCAL.h"

#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusteringParamsGPU.h"

class PFClusterProducerCudaHCAL : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  PFClusterProducerCudaHCAL(const edm::ParameterSet&);
  ~PFClusterProducerCudaHCAL() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  // the actual algorithm
  std::vector<std::unique_ptr<RecHitTopologicalCleanerBase>> _cleaners;
  std::unique_ptr<SeedFinderBase> _seedFinder;
  std::unique_ptr<InitialClusteringStepBase> _initialClustering;
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFCPositionCalculatorBase> _positionReCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPosCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _positionCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPositionCalc;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<cms::cuda::Product<hcal::PFRecHitCollection<pf::common::DevStoragePolicy>>> InputPFRecHitSoA_Token_;

  edm::ESGetToken<PFClusteringParamsGPU, JobConfigurationGPURecord> const pfClusParamsToken_;

  int nRH_ = 0;

  const bool _produceSoA;     // PFClusters in SoA format
  const bool _produceLegacy;  // PFClusters in legacy format

  // // options
  // const bool _prodInitClusters;

  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;

  //cms::cuda::ContextState cudaState_;

  PFClustering::HCAL::ConfigurationParameters cudaConfig_;

  PFClustering::HCAL::OutputDataCPU outputCPU;
  PFClustering::HCAL::OutputDataGPU outputGPU;

  PFClustering::HCAL::ScratchDataGPU scratchGPU;
};

PFClusterProducerCudaHCAL::PFClusterProducerCudaHCAL(const edm::ParameterSet& conf)
    : InputPFRecHitSoA_Token_{consumes(conf.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
      pfClusParamsToken_{esConsumes(conf.getParameter<edm::ESInputTag>("pfClusteringParameters"))},
      _produceSoA{conf.getParameter<bool>("produceSoA")},
      _produceLegacy{conf.getParameter<bool>("produceLegacy")},
      _rechitsLabel{consumes(conf.getParameter<edm::InputTag>("recHitsSource"))} {
  edm::ConsumesCollector cc = consumesCollector();

  //setup rechit cleaners
  const edm::VParameterSet& cleanerConfs = conf.getParameterSetVector("recHitCleaners");

  for (const auto& conf : cleanerConfs) {
    const std::string& cleanerName = conf.getParameter<std::string>("algoName");
    _cleaners.emplace_back(RecHitTopologicalCleanerFactory::get()->create(cleanerName, conf, cc));
  }

  // setup seed finding
  const edm::ParameterSet& sfConf = conf.getParameterSet("seedFinder");
  const std::string& sfName = sfConf.getParameter<std::string>("algoName");
  _seedFinder = SeedFinderFactory::get()->create(sfName, sfConf);

  const edm::VParameterSet& seedFinderConfs = sfConf.getParameterSetVector("thresholdsByDetector");

  //setup topo cluster builder
  const edm::ParameterSet& initConf = conf.getParameterSet("initialClusteringStep");
  const std::string& initName = initConf.getParameter<std::string>("algoName");
  _initialClustering = InitialClusteringStepFactory::get()->create(initName, initConf, cc);
  //setup pf cluster builder if requested
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");
  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf, cc);
    /*
    if (pfcConf.exists("allCellsPositionCalc")) {
    const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
    const std::string& algoac = acConf.getParameter<std::string>("algoName");
    _allCellsPosCalcCuda = PFCPositionCalculatorFactory::get()->create(algoac, acConf);
    */

    if (pfcConf.exists("positionCalc")) {
      const edm::ParameterSet& acConf = pfcConf.getParameterSet("positionCalc");
      const std::string& algoac = acConf.getParameter<std::string>("algoName");
      _positionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
    }

    if (pfcConf.exists("allCellsPositionCalc")) {
      const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
      const std::string& algoac = acConf.getParameter<std::string>("algoName");
      _allCellsPositionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
    }
  }
  //setup (possible) recalcuation of positions
  const edm::ParameterSet& pConf = conf.getParameterSet("positionReCalc");
  if (!pConf.empty()) {
    const std::string& pName = pConf.getParameter<std::string>("algoName");
    _positionReCalc = PFCPositionCalculatorFactory::get()->create(pName, pConf, cc);
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet& cConf = conf.getParameterSet("energyCorrector");
  if (!cConf.empty()) {
    const std::string& cName = cConf.getParameter<std::string>("algoName");
    _energyCorrector = PFClusterEnergyCorrectorFactory::get()->create(cName, cConf);
  }

  produces<reco::PFClusterCollection>();
}

PFClusterProducerCudaHCAL::~PFClusterProducerCudaHCAL() {}

void PFClusterProducerCudaHCAL::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("PFRecHitsLabelIn", edm::InputTag("hltParticleFlowRecHitHBHE"));
  desc.add<bool>("produceSoA", true);
  desc.add<bool>("produceLegacy", true);

  desc.add<edm::ESInputTag>("pfClusteringParameters",
                            edm::ESInputTag("pfClusteringParamsGPUESSource", "pfClusParamsOffline"));

  // Prevents the producer and navigator parameter sets from throwing an exception
  // TODO: Replace with a proper parameter set description: twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp
  desc.setAllowAnything();

  cdesc.addWithDefaultLabel(desc);
}

void PFClusterProducerCudaHCAL::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  //initCuda_ = true;  // (Re)initialize cuda arrays
  //KenH: for now comment this out, as we know we don't change the channel status on lumisection basis
}

void PFClusterProducerCudaHCAL::acquire(edm::Event const& event,
                                        edm::EventSetup const& setup,
                                        edm::WaitingTaskWithArenaHolder holder) {
  // Creates a new Cuda stream
  // TODO: Reuse stream from GPU PFRecHitProducer by passing input product as first arg
  // cmssdt.cern.ch/lxr/source/HeterogeneousCore/CUDACore/interface/ScopedContext.h#0101
  //cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};
  //cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder)};
  auto const& PFRecHitsProduct = event.get(InputPFRecHitSoA_Token_);
  cms::cuda::ScopedContextAcquire ctx{PFRecHitsProduct, std::move(holder)};
  auto const& PFRecHits = ctx.get(PFRecHitsProduct);
  auto cudaStream = ctx.stream();

  outputCPU.allocate(cudaConfig_, cudaStream);
  outputGPU.allocate(cudaConfig_, cudaStream);
  scratchGPU.allocate(cudaConfig_, cudaStream);

  nRH_ = PFRecHits.size;
  if (nRH_ == 0)
    return;
  if (nRH_ > 4000)
    std::cout << "nRH(PFRecHitSize)>4000: " << nRH_ << std::endl;

  const int numbytes_int = nRH_ * sizeof(int);
  int totalNeighbours = 0;  // Running count of 8 neighbour edges for edgeId, edgeList

  float kernelTimers[8] = {0.0};

  // if (cudaStreamQuery(cudaStream) != cudaSuccess)
  //   cudaCheck(cudaStreamSynchronize(cudaStream));

  auto const& pfClusParamsProduct = setup.getData(pfClusParamsToken_).getProduct(cudaStream);

  // Calling cuda kernels
  PFClusterCudaHCAL::PFRechitToPFCluster_HCAL_entryPoint(
      cudaStream, pfClusParamsProduct, totalNeighbours, PFRecHits, outputGPU, scratchGPU, kernelTimers);

  if (!_produceLegacy)
    return;  // do device->host transfer only when we are producing Legacy data

  // Data transfer from GPU
  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

  cudaCheck(cudaMemcpyAsync(
      outputCPU.pcrhFracSize.get(), outputGPU.pcrhFracSize.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));

  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

  // Total size of allocated rechit fraction arrays (includes some extra padding for rechits that don't end up passing cuts)
  const Int_t nFracs = outputCPU.pcrhFracSize[0];

  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoSeedCount.get(), outputGPU.topoSeedCount.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));

  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoRHCount.get(), outputGPU.topoRHCount.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));

  cudaCheck(cudaMemcpyAsync(outputCPU.seedFracOffsets.get(),
                            outputGPU.seedFracOffsets.get(),
                            numbytes_int,
                            cudaMemcpyDeviceToHost,
                            cudaStream));

  cudaCheck(cudaMemcpyAsync(outputCPU.pcrh_fracInd.get(),
                            outputGPU.pcrh_fracInd.get(),
                            sizeof(int) * nFracs,
                            cudaMemcpyDeviceToHost,
                            cudaStream));

  cudaCheck(cudaMemcpyAsync(
      outputCPU.pcrh_frac.get(), outputGPU.pcrh_frac.get(), sizeof(float) * nFracs, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfrh_isSeed.get(), outputGPU.pfrh_isSeed.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfrh_topoId.get(), outputGPU.pfrh_topoId.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));

  // if (cudaStreamQuery(cudaStream) != cudaSuccess)
  //   cudaCheck(cudaStreamSynchronize(cudaStream));
}

void PFClusterProducerCudaHCAL::produce(edm::Event& event, const edm::EventSetup& setup) {
  // cms::cuda::ScopedContextProduce ctx{cudaState_};
  // if (_produceSoA)
  //   ctx.emplace(event, OutputPFClusterSoA_Token_, std::move(outputGPU.PFClusters)); // SoA "PFClusters" still need to be defined.

  if (_produceLegacy) {
    auto pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();
    pfClustersFromCuda->reserve(nRH_);

    auto const rechitsHandle = event.getHandle(_rechitsLabel);

    // Build PFClusters in legacy format
    std::unordered_map<int, std::vector<int>> nTopoRechits;
    std::unordered_map<int, int> nTopoSeeds;

    for (int rh = 0; rh < nRH_; rh++) {
      int topoId = outputCPU.pfrh_topoId[rh];
      if (topoId > -1) {
        // Valid topo id
        nTopoRechits[topoId].push_back(rh);
        if (outputCPU.pfrh_isSeed[rh] > 0) {
          nTopoSeeds[topoId]++;
        }
      }
    }

    // Looping over PFRecHits for creating PFClusters
    for (int n = 0; n < nRH_; n++) {
      if (outputCPU.pfrh_isSeed[n] ==
          1) {  // If this PFRecHit is a seed, this should form a PFCluster. Compute necessary information.
        reco::PFCluster temp;
        temp.setSeed((*rechitsHandle)[n].detId());  // Pulling the detId of this PFRecHit from the legacy format input
        int offset = outputCPU.seedFracOffsets[n];
        int topoId = outputCPU.pfrh_topoId[n];
        int nSeeds = outputCPU.topoSeedCount[topoId];
        for (int k = offset; k < (offset + outputCPU.topoRHCount[topoId] - nSeeds + 1);
             k++) {  // Looping over PFRecHits in the same topo cluster
          if (outputCPU.pcrh_fracInd[k] > -1 && outputCPU.pcrh_frac[k] > 0.0) {
            const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechitsHandle, outputCPU.pcrh_fracInd[k]);
            temp.addRecHitFraction(reco::PFRecHitFraction(refhit, outputCPU.pcrh_frac[k]));
          }
        }
        // Now PFRecHitFraction of this PFCluster is set. Now compute calculateAndSetPosition (energy, position etc)
        // Check if this topoId has one only one seed
        if (nTopoSeeds.count(outputCPU.pfrh_topoId[n]) && nTopoSeeds[outputCPU.pfrh_topoId[n]] == 1 &&
            _allCellsPositionCalc) {
          _allCellsPositionCalc->calculateAndSetPosition(temp);
        } else {
          _positionCalc->calculateAndSetPosition(temp);
        }
        pfClustersFromCuda->emplace_back(std::move(temp));
      }
    }

    event.put(std::move(pfClustersFromCuda));
  }
}

DEFINE_FWK_MODULE(PFClusterProducerCudaHCAL);
