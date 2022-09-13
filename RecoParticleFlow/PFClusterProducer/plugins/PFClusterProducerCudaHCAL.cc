#include <memory>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "PFClusterCudaHCAL.h"
#include "PFClusterProducerCudaHCAL.h"

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

// Uncomment to enable GPU debugging
//#define DEBUG_GPU_HCAL

// Uncomment to fill TTrees
//#define DEBUG_HCAL_TREES

// Uncomment to save cluster collections in TTree
//#define DEBUG_SAVE_CLUSTERS

PFClusterProducerCudaHCAL::PFClusterProducerCudaHCAL(const edm::ParameterSet& conf)
    : InputPFRecHitSoA_Token_{consumes<IProductType>(conf.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
      _prodInitClusters(conf.getUntrackedParameter<bool>("prodInitialClusters", false)),
      _rechitsLabel{consumes<reco::PFRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsSource"))} {
  edm::ConsumesCollector cc = consumesCollector();

  nFracs_vs_nRH->GetXaxis()->SetTitle("nRH");
  nFracs_vs_nRH->GetYaxis()->SetTitle("nFracs");

  pfcIterations->GetXaxis()->SetTitle("PF clustering iterations");
  pfcIterations->GetYaxis()->SetTitle("Entries");

  pfcIter_vs_nRHTopo->GetXaxis()->SetTitle("Num rechits in topo cluster");
  pfcIter_vs_nRHTopo->GetYaxis()->SetTitle("PF clustering iterations");

  pfcIter_vs_nSeedsTopo->GetXaxis()->SetTitle("Num seeds in topo cluster");
  pfcIter_vs_nSeedsTopo->GetYaxis()->SetTitle("PF clustering iterations");

  pfcIter_vs_nFracsTopo->GetXaxis()->SetTitle("Num rechit fractions in topo cluster");
  pfcIter_vs_nFracsTopo->GetYaxis()->SetTitle("PF clustering iterations");

  topoIterations->GetXaxis()->SetTitle("Topo clustering iterations");
  topoIterations->GetYaxis()->SetTitle("Entries");

  topoIter_vs_nRH->GetXaxis()->SetTitle("Num rechits");
  topoIter_vs_nRH->GetYaxis()->SetTitle("Topo clustering iterations");

#ifdef DEBUG_HCAL_TREES
  hTimers->GetYaxis()->SetTitle("time (ms)");
  hTimers->GetXaxis()->SetBinLabel(1, "copyToDevice");
  hTimers->GetXaxis()->SetBinLabel(2, "seeding");
  hTimers->GetXaxis()->SetBinLabel(3, "topo");
  hTimers->GetXaxis()->SetBinLabel(4, "setup frac");
  hTimers->GetXaxis()->SetBinLabel(5, "PF clustering");
  hTimers->GetXaxis()->SetBinLabel(6, "copyToHost");

  //setup TTree
  clusterTree->Branch("Event", &numEvents);
  clusterTree->Branch("topoIter", &topoIter, "topoIter/I");
  clusterTree->Branch("nRH", &nRH, "nRH/I");
  clusterTree->Branch("nEdges", &nEdges, "nEdges/I");
  clusterTree->Branch("edgeId", &__edgeId);
  clusterTree->Branch("edgeList", &__edgeList);
  clusterTree->Branch("nFracs", &nFracs, "nFracs/I");
  clusterTree->Branch("nRHperPFCTotal_CPU", &nRHperPFCTotal_CPU, "nRHperPFCTotal_CPU/I");
  clusterTree->Branch("nRHperPFCTotal_GPU", &nRHperPFCTotal_GPU, "nRHperPFCTotal_GPU/I");
  clusterTree->Branch("timers", &GPU_timers);
  clusterTree->Branch("pfcIter", &__pfcIter);
  clusterTree->Branch("nRHTopo", &__nRHTopo);
  clusterTree->Branch("nSeedsTopo", &__nSeedsTopo);
  clusterTree->Branch("nFracsTopo", &__nFracsTopo);
  clusterTree->Branch("rechits", "PFRecHitCollection", &__rechits);
  clusterTree->Branch("rechits_isSeed", &__rh_isSeed);
  clusterTree->Branch("rechits_isSeedCPU", &__rh_isSeedCPU);
  clusterTree->Branch("rechits_x", &__rh_x);
  clusterTree->Branch("rechits_y", &__rh_y);
  clusterTree->Branch("rechits_z", &__rh_z);
  clusterTree->Branch("rechits_eta", &__rh_eta);
  clusterTree->Branch("rechits_phi", &__rh_phi);
  clusterTree->Branch("rechits_neighbours4", &__rh_neighbours4);
  clusterTree->Branch("rechits_neighbours8", &__rh_neighbours8);
  clusterTree->Branch("pfrh_detIdGPU", &__pfrh_detIdGPU);
  clusterTree->Branch("pfrh_neighboursGPU", &__pfrh_neighboursGPU);

#endif
#ifdef DEBUG_SAVE_CLUSTERS
  clusterTree->Branch("initialClusters", "PFClusterCollection", &__initialClusters);
  clusterTree->Branch("pfClusters", "PFClusterCollection", &__pfClusters);
  clusterTree->Branch("pfClustersFromCuda", "PFClusterCollection", &__pfClustersFromCuda);
#endif

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

  cudaConstants.minFracInCalc = 0.0;
  cudaConstants.minAllowedNormalization = 0.0;

  //setup topo cluster builder
  const edm::ParameterSet& initConf = conf.getParameterSet("initialClusteringStep");
  const std::string& initName = initConf.getParameter<std::string>("algoName");
  _initialClustering = InitialClusteringStepFactory::get()->create(initName, initConf, cc);
  //setup pf cluster builder if requested
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");
  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf, cc);
    /*if (pfcConf.exists("allCellsPositionCalc")) {
    const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
    const std::string& algoac = acConf.getParameter<std::string>("algoName");
    _allCellsPosCalcCuda = PFCPositionCalculatorFactory::get()->create(algoac, acConf);*/

    if (pfcConf.exists("positionCalc")) {
      const edm::ParameterSet& acConf = pfcConf.getParameterSet("positionCalc");
      const std::string& algoac = acConf.getParameter<std::string>("algoName");
      _positionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
      cudaConstants.minFracInCalc = (float)acConf.getParameter<double>("minFractionInCalc");
      cudaConstants.minAllowedNormalization = (float)acConf.getParameter<double>("minAllowedNormalization");
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

  cudaConstants.showerSigma2 = (float)std::pow(pfcConf.getParameter<double>("showerSigma"), 2.);
  const auto& recHitEnergyNormConf = pfcConf.getParameterSetVector("recHitEnergyNorms");
  for (const auto& pset : recHitEnergyNormConf) {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& recHitENorms = pset.getParameter<std::vector<double>>("recHitEnergyNorm");
      std::copy(recHitENorms.begin(), recHitENorms.end(), cudaConstants.recHitEnergyNormInvEB_vec);
      for (auto& x : cudaConstants.recHitEnergyNormInvEB_vec)
        x = std::pow(x, -1);  // Invert these values
    } else if (det == std::string("HCAL_ENDCAP")) {
      const auto& recHitENorms = pset.getParameter<std::vector<double>>("recHitEnergyNorm");
      std::copy(recHitENorms.begin(), recHitENorms.end(), cudaConstants.recHitEnergyNormInvEE_vec);
      for (auto& x : cudaConstants.recHitEnergyNormInvEE_vec)
        x = std::pow(x, -1);  // Invert these values
    } else
      std::cout << "Unknown detector when parsing recHitEnergyNorm: " << det << std::endl;
  }
  //float recHitEnergyNormEB = 0.08;
  //float recHitEnergyNormEE = 0.3;
  //float minFracToKeep = 0.0000001;
  cudaConstants.minFracToKeep = (float)pfcConf.getParameter<double>("minFractionToKeep");
  cudaConstants.minFracTot = (float)pfcConf.getParameter<double>("minFracTot");

  // Max PFClustering iterations
  cudaConstants.maxIterations = pfcConf.getParameter<unsigned>("maxIterations");

  cudaConstants.excludeOtherSeeds = pfcConf.getParameter<bool>("excludeOtherSeeds");

  //float stoppingTolerance2 = (float)std::pow(pfcConf.getParameter<double>("stoppingTolerance"), 2.0);
  cudaConstants.stoppingTolerance = (float)pfcConf.getParameter<double>("stoppingTolerance");

  cudaConstants.seedPt2ThresholdEB = -1;
  cudaConstants.seedPt2ThresholdEE = -1;
  for (const auto& pset : seedFinderConfs) {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& thresholds = pset.getParameter<std::vector<double>>("seedingThreshold");
      std::copy(thresholds.begin(), thresholds.end(), cudaConstants.seedEThresholdEB_vec);
      cudaConstants.seedPt2ThresholdEB =
          (float)std::pow(pset.getParameter<std::vector<double>>("seedingThresholdPt")[0], 2.0);

    } else if (det == std::string("HCAL_ENDCAP")) {
      const auto& thresholds = pset.getParameter<std::vector<double>>("seedingThreshold");
      std::copy(thresholds.begin(), thresholds.end(), cudaConstants.seedEThresholdEE_vec);
      cudaConstants.seedPt2ThresholdEE =
          (float)std::pow(pset.getParameter<std::vector<double>>("seedingThresholdPt")[0], 2.0);
    } else
      std::cout << "Unknown detector when parsing seedFinder: " << det << std::endl;
  }

  const auto& topoThresholdConf = initConf.getParameterSetVector("thresholdsByDetector");
  for (const auto& pset : topoThresholdConf) {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& thresholds = pset.getParameter<std::vector<double>>("gatheringThreshold");
      std::copy(thresholds.begin(), thresholds.end(), cudaConstants.topoEThresholdEB_vec);
    } else if (det == std::string("HCAL_ENDCAP")) {
      const auto& thresholds = pset.getParameter<std::vector<double>>("gatheringThreshold");
      std::copy(thresholds.begin(), thresholds.end(), cudaConstants.topoEThresholdEE_vec);
    } else
      std::cout << "Unknown detector when parsing initClusteringStep: " << det << std::endl;
  }

  if (pfcConf.exists("timeResolutionCalcEndcap")) {
    const edm::ParameterSet& endcapTimeResConf = pfcConf.getParameterSet("timeResolutionCalcEndcap");
    cudaConstants.endcapTimeResConsts.corrTermLowE = (float)endcapTimeResConf.getParameter<double>("corrTermLowE");
    cudaConstants.endcapTimeResConsts.threshLowE = (float)endcapTimeResConf.getParameter<double>("threshLowE");
    cudaConstants.endcapTimeResConsts.noiseTerm = (float)endcapTimeResConf.getParameter<double>("noiseTerm");
    cudaConstants.endcapTimeResConsts.constantTermLowE2 =
        (float)std::pow(endcapTimeResConf.getParameter<double>("constantTermLowE"), 2.0);
    cudaConstants.endcapTimeResConsts.noiseTermLowE = (float)endcapTimeResConf.getParameter<double>("noiseTermLowE");
    cudaConstants.endcapTimeResConsts.threshHighE = (float)endcapTimeResConf.getParameter<double>("threshHighE");
    cudaConstants.endcapTimeResConsts.constantTerm2 =
        (float)std::pow(endcapTimeResConf.getParameter<double>("constantTerm"), 2.0);
    cudaConstants.endcapTimeResConsts.resHighE2 =
        (float)std::pow(cudaConstants.endcapTimeResConsts.noiseTerm / cudaConstants.endcapTimeResConsts.threshHighE,
                        2.0) +
        cudaConstants.endcapTimeResConsts.constantTerm2;
  }

  if (pfcConf.exists("timeResolutionCalcBarrel")) {
    const edm::ParameterSet& barrelTimeResConf = pfcConf.getParameterSet("timeResolutionCalcBarrel");
    cudaConstants.barrelTimeResConsts.corrTermLowE = (float)barrelTimeResConf.getParameter<double>("corrTermLowE");
    cudaConstants.barrelTimeResConsts.threshLowE = (float)barrelTimeResConf.getParameter<double>("threshLowE");
    cudaConstants.barrelTimeResConsts.noiseTerm = (float)barrelTimeResConf.getParameter<double>("noiseTerm");
    cudaConstants.barrelTimeResConsts.constantTermLowE2 =
        (float)std::pow(barrelTimeResConf.getParameter<double>("constantTermLowE"), 2.0);
    cudaConstants.barrelTimeResConsts.noiseTermLowE = (float)barrelTimeResConf.getParameter<double>("noiseTermLowE");
    cudaConstants.barrelTimeResConsts.threshHighE = (float)barrelTimeResConf.getParameter<double>("threshHighE");
    cudaConstants.barrelTimeResConsts.constantTerm2 =
        (float)std::pow(barrelTimeResConf.getParameter<double>("constantTerm"), 2.0);
    cudaConstants.barrelTimeResConsts.resHighE2 =
        (float)std::pow(cudaConstants.barrelTimeResConsts.noiseTerm / cudaConstants.barrelTimeResConsts.threshHighE,
                        2.0) +
        cudaConstants.barrelTimeResConsts.constantTerm2;
  }
  cudaConstants.nNeigh = sfConf.getParameter<int>("nNeighbours");

  pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();

  if (_prodInitClusters) {
    produces<reco::PFClusterCollection>("initialClusters");
  }
  produces<reco::PFClusterCollection>();
}

PFClusterProducerCudaHCAL::~PFClusterProducerCudaHCAL() {
  // Free Cuda memory
  freeCudaMemory();

  MyFile->cd();
#if defined DEBUG_HCAL_TREES || defined DEBUG_SAVE_CLUSTERS
  clusterTree->Write();
#endif
  nFracs_vs_nRH->Write();
  pfcIterations->Write();
  pfcIter_vs_nRHTopo->Write();
  pfcIter_vs_nSeedsTopo->Write();
  pfcIter_vs_nFracsTopo->Write();
  topoIterations->Write();
  topoIter_vs_nRH->Write();
  nTopo_CPU->Write();
  nTopo_GPU->Write();
  topoSeeds_CPU->Write();
  topoSeeds_GPU->Write();
  sumSeed_CPU->Write();
  sumSeed_GPU->Write();
  topoEn_CPU->Write();
  topoEn_GPU->Write();
  topoEta_CPU->Write();
  topoEta_GPU->Write();
  topoPhi_CPU->Write();
  topoPhi_GPU->Write();
  nPFCluster_CPU->Write();
  nPFCluster_GPU->Write();
  enPFCluster_CPU->Write();
  enPFCluster_GPU->Write();
  pfcEta_CPU->Write();
  pfcEta_GPU->Write();
  pfcPhi_CPU->Write();
  pfcPhi_GPU->Write();
  nRH_perPFCluster_CPU->Write();
  nRH_perPFCluster_GPU->Write();
  nRH_perPFClusterTotal_CPU->Write();
  nRH_perPFClusterTotal_GPU->Write();
  matched_pfcRh_CPU->Write();
  matched_pfcRh_GPU->Write();
  matched_pfcEn_CPU->Write();
  matched_pfcEn_GPU->Write();
  matched_pfcEta_CPU->Write();
  matched_pfcEta_GPU->Write();
  matched_pfcPhi_CPU->Write();
  matched_pfcPhi_GPU->Write();
  nRh_CPUvsGPU->Write();
  enPFCluster_CPUvsGPU->Write();
  enPFCluster_CPUvsGPU_1d->Write();
  coordinate->Write();
  layer->Write();
  deltaSumSeed->Write();
  deltaRH->Write();
  deltaEn->Write();
  deltaEta->Write();
  deltaPhi->Write();

  if (doComparison) {
    std::cout << "\n--- CPU/GPU Comparison plots ---\nPlot\t\tEntries\tMean\t\tRMS\n\n"
              << "deltaRH\t\t" << deltaRH->GetEntries() << "\t" << deltaRH->GetMean()
              << ((deltaRH->GetMean() == 0) ? "\t\t" : "\t") << deltaRH->GetRMS() << std::endl
              << "deltaEn\t\t" << deltaEn->GetEntries() << "\t" << deltaEn->GetMean() << "\t" << deltaEn->GetRMS()
              << std::endl
              << "deltaEta\t" << deltaEta->GetEntries() << "\t" << deltaEta->GetMean() << "\t" << deltaEta->GetRMS()
              << std::endl
              << "deltaPhi\t" << deltaPhi->GetEntries() << "\t" << deltaPhi->GetMean() << "\t" << deltaPhi->GetRMS()
              << std::endl
              << std::endl;
  }

  if (numEvents > 10) {
    // Skip first 10 entries
    hTimers->Scale(1. / (numEvents - 10.));
  }
  hTimers->Write();
  delete MyFile;
}

void PFClusterProducerCudaHCAL::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("PFRecHitsLabelIn", edm::InputTag("hltParticleFlowRecHitHBHE"));
  // Prevents the producer and navigator parameter sets from throwing an exception
  // TODO: Replace with a proper parameter set description: twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp
  desc.setAllowAnything();

  cdesc.addWithDefaultLabel(desc);
}

void PFClusterProducerCudaHCAL::initializeCudaMemory(cudaStream_t cudaStream) {
  //  cudaCheck(cudaMallocManaged(&cuda_pcrhFracSize, sizeof(int)));
  //  cudaCheck(cudaMallocManaged(&cuda_topoIter, sizeof(int)));
}

void PFClusterProducerCudaHCAL::freeCudaMemory() {
  //  cudaCheck(cudaFree(cuda_pcrhFracSize));
  //  cudaCheck(cudaFree(cuda_topoIter));
}

void PFClusterProducerCudaHCAL::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& es) {
  _initialClustering->update(es);
  if (_pfClusterBuilder)
    _pfClusterBuilder->update(es);
  if (_positionReCalc)
    _positionReCalc->update(es);
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
  cudaStream = ctx.stream();
  nRH = PFRecHits.size;

#ifdef DEBUG_HCAL_TREES
  std::cout << "Found GPU PFRecHits with size = " << PFRecHits.size << "\tcleaned size = " << PFRecHits.sizeCleaned
            << std::endl;

  hcal::PFRecHitCollection<pf::common::VecStoragePolicy<pf::common::CUDAHostAllocatorAlias>> tmpPFRecHits;
  tmpPFRecHits.resize(PFRecHits.size);

  auto lambdaToTransferSize = [&ctx](auto& dest, auto* src, auto size) {
    using vector_type = typename std::remove_reference<decltype(dest)>::type;
    using src_data_type = typename std::remove_pointer<decltype(src)>::type;
    using type = typename vector_type::value_type;
    static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
    cudaCheck(cudaMemcpyAsync(dest.data(), src, size * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };

  lambdaToTransferSize(tmpPFRecHits.pfrh_detId, PFRecHits.pfrh_detId.get(), PFRecHits.size);
  lambdaToTransferSize(tmpPFRecHits.pfrh_neighbours, PFRecHits.pfrh_neighbours.get(), 8 * PFRecHits.size);
  if (cudaStreamQuery(ctx.stream()) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(ctx.stream()));
#endif

  if (numEvents == 0) {
    // Only allocate Cuda memory on first event
    PFClusterCudaHCAL::initializeCudaConstants(cudaConstants, cudaStream);

    initializeCudaMemory(cudaStream);

    inputCPU.allocate(cudaConfig_, cudaStream);
    inputGPU.allocate(cudaConfig_, cudaStream);

    outputCPU.allocate(cudaConfig_, cudaStream);
    outputGPU.allocate(cudaConfig_, cudaStream);
    scratchGPU.allocate(cudaConfig_, cudaStream);
  }

  _initialClustering->reset();
  if (_pfClusterBuilder)
    _pfClusterBuilder->reset();

  edm::Handle<reco::PFRecHitCollection> rechits;
  event.getByToken(_rechitsLabel, rechits);
  //  std::cout<<"\n===== Now on event "<<numEvents<<" with "<<nRH<<" HCAL rechits ====="<<std::endl;

#ifdef DEBUG_HCAL_TREES
  GPU_timers.fill(0.0);
  __pfcIter.clear();
  __nRHTopo.clear();
  __nSeedsTopo.clear();
  __nFracsTopo.clear();
  __rechits = *rechits;
  __rh_isSeed.clear();
  __rh_isSeedCPU.clear();
  __rh_x.clear();
  __rh_y.clear();
  __rh_z.clear();
  __rh_eta.clear();
  __rh_phi.clear();
  __rh_neighbours4.clear();
  __rh_neighbours8.clear();
  __edgeId.clear();
  __edgeList.clear();
  __pfrh_detIdGPU.clear();
  __pfrh_neighboursGPU.clear();

  __pfrh_detIdGPU.reserve(PFRecHits.size);
  __pfrh_neighboursGPU.reserve(8 * PFRecHits.size);
  for (int i = 0; i < (int)PFRecHits.size; i++) {
    __pfrh_detIdGPU.push_back(tmpPFRecHits.pfrh_detId[i]);
    for (int n = 0; n < 8; n++) {
      int neighIdx = tmpPFRecHits.pfrh_neighbours[8 * i + n];
      __pfrh_neighboursGPU.push_back(neighIdx);
      if (neighIdx >= (int)PFRecHits.size)
        std::cout << "*** WARNING: Invalid neighbor index for rechit " << i << " (" << tmpPFRecHits.pfrh_detId[i]
                  << ") neighbor " << n << ": " << neighIdx << " ***" << std::endl;
      int neighDetId = neighIdx > -1 ? tmpPFRecHits.pfrh_detId[neighIdx] : -1;
      __pfrh_neighboursDetIdGPU.push_back(neighDetId);
    }
  }
#endif

  _initialClustering->updateEvent(event);

#ifdef DEBUG_GPU_HCAL
  int numbytes_float = (int)rechits->size() * sizeof(float);
#endif
  int numbytes_int = (int)rechits->size() * sizeof(int);
  int totalNeighbours = 0;  // Running count of 8 neighbour edges for edgeId, edgeList

  int p = 0;

  //  std::cout<<"-----------------------------------------"<<std::endl;
  //  std::cout<<" HCAL: Event "<<numEvents<<" has "<<nRH<<" rechits"<<std::endl;
  //  std::cout<<"-----------------------------------------"<<std::endl;

  for (const auto& rh : *rechits) {
    //    if (p >= 10) break;
    inputCPU.pfrh_x[p] = rh.position().x();
    inputCPU.pfrh_y[p] = rh.position().y();
    inputCPU.pfrh_z[p] = rh.position().z();
    inputCPU.pfrh_energy[p] = rh.energy();
    inputCPU.pfrh_layer[p] = (int)rh.layer();
    inputCPU.pfrh_depth[p] = (int)rh.depth();

    auto theneighboursEight = rh.neighbours8();
    auto theneighboursFour = rh.neighbours4();
#ifdef DEBUG_HCAL_TREES
    __rh_x.push_back(inputCPU.pfrh_x[p]);
    __rh_y.push_back(inputCPU.pfrh_y[p]);
    __rh_z.push_back(inputCPU.pfrh_z[p]);
    __rh_eta.push_back(rh.positionREP().eta());
    __rh_phi.push_back(rh.positionREP().phi());
#endif
    std::vector<int> n4;
    std::vector<int> n8;
    for (auto nh : theneighboursEight) {
      n8.push_back((int)nh);
    }
    std::sort(n8.begin(), n8.end());  // Sort 8 neighbour edges in ascending order for topo clustering

    std::unordered_map<int, int> duplicates;
    for (int i = 0; i < 8; i++) {
      if (i < (int)n8.size()) {
        int nh = (int)n8[i];
        duplicates[nh]++;
        inputCPU.pfNeighEightInd[8 * p + i] = nh;
        inputCPU.pfrh_edgeId[totalNeighbours] = p;
        inputCPU.pfrh_edgeList[totalNeighbours] = nh;
        __edgeId.push_back(p);
        __edgeList.push_back(nh);
        totalNeighbours++;
      } else {
        inputCPU.pfNeighEightInd[8 * p + i] = -1;
      }
    }

    for (auto nh : theneighboursFour) {
      n4.push_back((int)nh);
    }
    std::sort(n4.begin(), n4.end());

    for (int i = 0; i < 4; i++) {
      if (i < (int)n4.size()) {
        inputCPU.pfNeighFourInd[4 * p + i] = (int)n4[i];
      } else {
        inputCPU.pfNeighFourInd[4 * p + i] = -1;
      }
    }

    p++;
#ifdef DEBUG_HCAL_TREES
    __rh_neighbours4.push_back(n4);
    __rh_neighbours8.push_back(n8);
#endif
  }  //end of rechit loop

#ifdef DEBUG_GPU_HCAL
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, cudaStream);
#endif

#ifdef DEBUG_GPU_HCAL
  // Copy arrays to GPU memory
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_x.get(), inputCPU.pfrh_x.get(), numbytes_float, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_y.get(), inputCPU.pfrh_y.get(), numbytes_float, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_z.get(), inputCPU.pfrh_z.get(), numbytes_float, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_energy.get(), inputCPU.pfrh_energy.get(), numbytes_float, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_layer.get(), inputCPU.pfrh_layer.get(), numbytes_int, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      inputGPU.pfrh_depth.get(), inputCPU.pfrh_depth.get(), numbytes_int, cudaMemcpyHostToDevice, cudaStream));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfNeighEightInd.get(),
                            inputCPU.pfNeighEightInd.get(),
                            numbytes_int * 8,
                            cudaMemcpyHostToDevice,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfNeighFourInd.get(),
                            inputCPU.pfNeighFourInd.get(),
                            numbytes_int * 4,
                            cudaMemcpyHostToDevice,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_edgeId.get(),
                            inputCPU.pfrh_edgeId.get(),
                            sizeof(int) * totalNeighbours,
                            cudaMemcpyHostToDevice,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(inputGPU.pfrh_edgeList.get(),
                            inputCPU.pfrh_edgeList.get(),
                            sizeof(int) * totalNeighbours,
                            cudaMemcpyHostToDevice,
                            cudaStream));
#endif

#ifdef DEBUG_GPU_HCAL
  cudaEventRecord(stop, cudaStream);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPU_timers[0], start, stop);
  //std::cout<<"(HCAL) Copy memory to device: "<<GPU_timers[0]<<" ms"<<std::endl;
//  cudaEventRecord(start);
#endif

  float kernelTimers[8] = {0.0};

  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

  PFClusterCudaHCAL::PFRechitToPFCluster_HCAL_entryPoint(
      cudaStream, (int)totalNeighbours, PFRecHits, inputGPU, outputCPU, outputGPU, scratchGPU, kernelTimers);

  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoIter.get(), outputGPU.topoIter.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pcrhFracSize.get(), outputGPU.pcrhFracSize.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(
      cudaMemcpyAsync(outputCPU.nEdges.get(), outputGPU.nEdges.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

  // Total size of allocated rechit fraction arrays (includes some extra padding for rechits that don't end up passing cuts)
  nFracs = outputCPU.pcrhFracSize[0];
  nEdges = outputCPU.nEdges[0];
#ifdef DEBUG_GPU_HCAL
  GPU_timers[1] = kernelTimers[0];  // Seeding
  GPU_timers[2] = kernelTimers[1];  // Topo clustering
  GPU_timers[3] = kernelTimers[2];
  GPU_timers[4] = kernelTimers[3];  // PF clustering

  // Extra timers
  GPU_timers[6] = kernelTimers[4];
  GPU_timers[7] = kernelTimers[5];
  GPU_timers[8] = kernelTimers[6];

  //  std::cout<<"HCAL GPU clustering (ms):\n"
  //           <<"Seeding\t\t"<<GPU_timers[1]<<std::endl
  //           <<"Topo clustering\t"<<GPU_timers[2]<<std::endl
  //           <<"PF cluster step 1 \t"<<GPU_timers[3]<<std::endl
  //           <<"PF cluster step 2 \t"<<GPU_timers[4]<<std::endl;
  cudaEventRecord(start, cudaStream);
#endif

  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfc_iter.get(), outputGPU.pfc_iter.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoSeedCount.get(), outputGPU.topoSeedCount.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoRHCount.get(), outputGPU.topoRHCount.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(outputCPU.seedFracOffsets.get(),
                            outputGPU.seedFracOffsets.get(),
                            numbytes_int,
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(outputCPU.topoSeedOffsets.get(),
                            outputGPU.topoSeedOffsets.get(),
                            numbytes_int,
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoSeedList.get(), outputGPU.topoSeedList.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));

  cudaCheck(cudaMemcpyAsync(outputCPU.pcrh_fracInd.get(),
                            outputGPU.pcrh_fracInd.get(),
                            sizeof(int) * nFracs,
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pcrh_frac.get(), outputGPU.pcrh_frac.get(), sizeof(int) * nFracs, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfrh_isSeed.get(), outputGPU.pfrh_isSeed.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfrh_topoId.get(), outputGPU.pfrh_topoId.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));

  cudaCheck(cudaMemcpyAsync(outputCPU.pfrh_passTopoThresh.get(),
                            outputGPU.pfrh_passTopoThresh.get(),
                            sizeof(int) * nRH,
                            cudaMemcpyDeviceToHost,
                            cudaStream));

#ifdef DEBUG_GPU_HCAL
  cudaEventRecord(stop, cudaStream);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&GPU_timers[5], start, stop);
//  std::cout<<"(HCAL) Copy results from GPU: "<<GPU_timers[5]<<" ms"<<std::endl;
#endif
  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

#ifdef DEBUG_GPU_HCAL
  nFracs_vs_nRH->Fill(nRH, nFracs);
  for (int topoId = 0; topoId < (int)nRH; topoId++) {
    int nPFCIter = outputCPU.pfc_iter[topoId];  // Number of iterations for PF clustering to converge
    int nSeeds = outputCPU.topoSeedCount[topoId];
    int nRHTopo = outputCPU.topoRHCount[topoId];
    int nFracsTopo = (nRHTopo - nSeeds + 1) * nSeeds;
#ifdef DEBUG_HCAL_TREES
    __pfcIter.push_back(nPFCIter);
    __nRHTopo.push_back(nRHTopo);
    __nSeedsTopo.push_back(nSeeds);
    __nFracsTopo.push_back(nFracsTopo);
#endif
    if (nPFCIter >= 0) {
      pfcIterations->Fill(nPFCIter);
      pfcIter_vs_nRHTopo->Fill(nRHTopo, nPFCIter);
      pfcIter_vs_nSeedsTopo->Fill(nSeeds, nPFCIter);
      pfcIter_vs_nFracsTopo->Fill(nFracsTopo, nPFCIter);  // Total number of rechit fractions for a given topo cluster
    }
  }
#endif

  std::unordered_map<int, std::vector<int>> nTopoRechits;
  std::unordered_map<int, int> nTopoSeeds;

  for (int rh = 0; rh < nRH; rh++) {
    int topoId = outputCPU.pfrh_topoId[rh];
    if (topoId > -1) {
      // Valid topo id
      nTopoRechits[topoId].push_back(rh);
      if (outputCPU.pfrh_isSeed[rh] > 0) {
        nTopoSeeds[topoId]++;
      }
    }
  }

  int intTopoCount = 0;
  for (const auto& x : nTopoRechits) {
    int topoId = x.first;
    if (nTopoSeeds.count(topoId) > 0) {
      // This topo cluster has at least one seed
      nTopo_GPU->Fill(x.second.size());
      topoSeeds_GPU->Fill(nTopoSeeds[topoId]);
      intTopoCount++;
    }
  }

  nPFCluster_GPU->Fill(intTopoCount);

  if (doComparison) {
    for (int i = 0; i < nRH; i++) {
      int topoIda = outputCPU.pfrh_topoId[i];
      if (nTopoSeeds.count(topoIda) == 0)
        continue;
      for (int j = 0; j < 8; j++) {
        if (inputCPU.pfNeighEightInd[i * 8 + j] > -1 &&
            outputCPU.pfrh_topoId[inputCPU.pfNeighEightInd[i * 8 + j]] != topoIda &&
            outputCPU.pfrh_passTopoThresh[i * 8 + j])
          std::cout << "HCAL HAS DIFFERENT TOPOID " << i << "  " << j << "  " << topoIda << "  "
                    << outputCPU.pfrh_topoId[inputCPU.pfNeighEightInd[i * 8 + j]] << std::endl;
      }
    }
  }

  topoIter = outputCPU.topoIter[0];
  topoIterations->Fill(topoIter);
  topoIter_vs_nRH->Fill(nRH, topoIter);

  //auto pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();
  pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();

  for (int n = 0; n < (int)nRH; n++) {
#ifdef DEBUG_HCAL_TREES
    __rh_isSeed.push_back(outputCPU.pfrh_isSeed[n]);
#endif
    if (outputCPU.pfrh_isSeed[n] == 1) {
      reco::PFCluster temp;
      temp.setSeed((*rechits)[n].detId());
      int offset = outputCPU.seedFracOffsets[n];
      int topoId = outputCPU.pfrh_topoId[n];
      int nSeeds = outputCPU.topoSeedCount[topoId];
      for (int k = offset; k < (offset + outputCPU.topoRHCount[topoId] - nSeeds + 1); k++) {
        if (outputCPU.pcrh_fracInd[k] > -1 && outputCPU.pcrh_frac[k] > 0.0) {
          const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechits, outputCPU.pcrh_fracInd[k]);
          temp.addRecHitFraction(reco::PFRecHitFraction(refhit, outputCPU.pcrh_frac[k]));
        }
      }
      // Check if this topoId has one only one seed
      if (nTopoSeeds.count(outputCPU.pfrh_topoId[n]) && nTopoSeeds[outputCPU.pfrh_topoId[n]] == 1 &&
          _allCellsPositionCalc) {
        _allCellsPositionCalc->calculateAndSetPosition(temp);
      } else {
        _positionCalc->calculateAndSetPosition(temp);
      }
      pfClustersFromCuda->insert(pfClustersFromCuda->end(), std::move(temp));
    }
  }

#ifdef DEBUG_SAVE_CLUSTERS
  __pfClustersFromCuda = *pfClustersFromCuda;  // For TTree
#endif

  //if (_energyCorrector) {
  //  _energyCorrector->correctEnergies(*pfClustersFromCuda);
  //}

  float sumEn_CPU = 0.;
  if (doComparison) {
    std::vector<bool> mask(nRH, true);
    std::vector<bool> seedable(nRH, false);
    _seedFinder->findSeeds(rechits, mask, seedable);
#ifdef DEBUG_HCAL_TREES
    __rh_isSeedCPU.resize(nRH);
    for (int i = 0; i < (int)seedable.size(); i++)
      __rh_isSeedCPU[i] = (int)seedable[i];
#endif

    auto initialClusters = std::make_unique<reco::PFClusterCollection>();
    _initialClustering->buildClusters(rechits, mask, seedable, *initialClusters);
#ifdef DEBUG_SAVE_CLUSTERS
    __initialClusters = *initialClusters;  // For TTree
#endif
    int topoRhCount = 0;
    for (const auto& pfc : *initialClusters) {
      nTopo_CPU->Fill(pfc.recHitFractions().size());
      topoEn_CPU->Fill(pfc.energy());
      topoEta_CPU->Fill(pfc.eta());
      topoPhi_CPU->Fill(pfc.phi());
      topoRhCount = topoRhCount + pfc.recHitFractions().size();
      int nSeeds = 0;
      for (const auto& rhf : pfc.recHitFractions()) {
        if (seedable[rhf.recHitRef().key()])
          nSeeds++;
      }
      topoSeeds_CPU->Fill(nSeeds);
    }

    nPFCluster_CPU->Fill(initialClusters->size());

    LOGVERB("PFClusterProducer::produce()") << *_initialClustering;

    int seedSumCPU = 0;
    int seedSumGPU = 0;
    int maskSize = 0;
    for (int j = 0; j < (int)seedable.size(); j++)
      seedSumCPU = seedSumCPU + seedable[j];
    for (int j = 0; j < (int)nRH; j++)
      seedSumGPU = seedSumGPU + outputCPU.pfrh_isSeed[j];
    for (int j = 0; j < (int)mask.size(); j++)
      maskSize = maskSize + mask[j];

    sumSeed_CPU->Fill(seedSumCPU);
    sumSeed_GPU->Fill(seedSumGPU);
    deltaSumSeed->Fill(seedSumGPU - seedSumCPU);

    pfClusters = std::make_unique<reco::PFClusterCollection>();
    pfClusters = std::make_unique<reco::PFClusterCollection>();
    if (_pfClusterBuilder) {  // if we've defined a re-clustering step execute it
      _pfClusterBuilder->buildClusters(*initialClusters, seedable, *pfClusters);
      LOGVERB("PFClusterProducer::produce()") << *_pfClusterBuilder;
    } else {
      pfClusters->insert(pfClusters->end(), initialClusters->begin(), initialClusters->end());
    }

    int totalRHPF_CPU = 0, totalRHPF_GPU = 0;
#ifdef DEBUG_SAVE_CLUSTERS
    __pfClusters = *pfClusters;  // For TTree
#endif
    for (const auto& pfc : *pfClusters) {
      nRH_perPFCluster_CPU->Fill(pfc.recHitFractions().size());
      totalRHPF_CPU += (int)pfc.recHitFractions().size();
      enPFCluster_CPU->Fill(pfc.energy());
      pfcEta_CPU->Fill(pfc.eta());
      pfcPhi_CPU->Fill(pfc.phi());
      sumEn_CPU += pfc.energy();
      for (const auto& pfcx : *pfClustersFromCuda) {
        if (pfc.seed() == pfcx.seed()) {
          totalRHPF_GPU += (int)pfcx.recHitFractions().size();

          matched_pfcRh_CPU->Fill(pfc.recHitFractions().size());
          matched_pfcRh_GPU->Fill(pfcx.recHitFractions().size());
          matched_pfcEn_CPU->Fill(pfc.energy());
          matched_pfcEn_GPU->Fill(pfcx.energy());
          matched_pfcEta_CPU->Fill(pfc.eta());
          matched_pfcEta_GPU->Fill(pfcx.eta());
          matched_pfcPhi_CPU->Fill(pfc.phi());
          matched_pfcPhi_GPU->Fill(pfcx.phi());

          if (abs((int)pfcx.recHitFractions().size() - (int)pfc.recHitFractions().size()) > 0) {
            std::cout << "HCAL mismatch nRH:\tGPU:" << (int)pfcx.recHitFractions().size()
                      << "\tCPU:" << (int)pfc.recHitFractions().size() << std::endl;
          }
          deltaRH->Fill((int)pfcx.recHitFractions().size() - (int)pfc.recHitFractions().size());
          if (abs(pfcx.energy() - pfc.energy()) > 1e-2) {
            std::cout << "HCAL mismatch  En:\tGPU:" << pfcx.energy() << "\tCPU:" << pfc.energy() << std::endl;
          }
          deltaEn->Fill(pfcx.energy() - pfc.energy());
          if (abs(pfcx.eta() - pfc.eta()) > 1e-4) {
            std::cout << "HCAL mismatch Eta:\tGPU:" << pfcx.eta() << "\tCPU:" << pfc.eta() << std::endl;
          }
          deltaEta->Fill(pfcx.eta() - pfc.eta());
          if (abs(pfcx.phi() - pfc.phi()) > 1e-4) {
            std::cout << "HCAL mismatch Phi:\tGPU:" << pfcx.phi() << "\tCPU:" << pfc.phi() << std::endl;
          }
          deltaPhi->Fill(pfcx.phi() - pfc.phi());

          nRh_CPUvsGPU->Fill(pfcx.recHitFractions().size(), pfc.recHitFractions().size());
          enPFCluster_CPUvsGPU->Fill(pfcx.energy(), pfc.energy());
          enPFCluster_CPUvsGPU_1d->Fill((pfcx.energy() - pfc.energy()) / pfc.energy());
          if (abs((pfcx.energy() - pfc.energy()) / pfc.energy()) > 0.05) {
            coordinate->Fill(pfcx.eta(), pfcx.phi());

            for (const auto& rhf : pfc.recHitFractions()) {
              if (rhf.fraction() == 1)
                layer->Fill(rhf.recHitRef()->depth());
            }
          }
        }
      }
    }

    nRH_perPFClusterTotal_CPU->Fill(totalRHPF_CPU);
    nRH_perPFClusterTotal_GPU->Fill(totalRHPF_GPU);

    nRHperPFCTotal_CPU = totalRHPF_CPU;
    nRHperPFCTotal_GPU = totalRHPF_GPU;

    for (const auto& pfc : *pfClustersFromCuda) {
      nRH_perPFCluster_GPU->Fill(pfc.recHitFractions().size());
      enPFCluster_GPU->Fill(pfc.energy());
      pfcEta_GPU->Fill(pfc.eta());
      pfcPhi_GPU->Fill(pfc.phi());
    }
  }

#ifdef DEBUG_GPU_HCAL
  if (numEvents > 9) {
    for (int i = 0; i < (int)GPU_timers.size(); i++)
      hTimers->Fill(i, GPU_timers[i]);
  }
#endif

#if defined DEBUG_HCAL_TREES || defined DEBUG_SAVE_CLUSTERS
  clusterTree->Fill();
#endif
  if (doComparison)
    std::cout << "For " << nRH << " input PFRecHits, found nEdges = " << nEdges << "  nFracs = " << nFracs
              << "  (CPU) pfClusters->size() = " << (int)pfClusters->size()
              << "  (GPU) pfClustersFromCuda->size() = " << (int)pfClustersFromCuda->size() << std::endl
              << std::endl;
  else
    std::cout << "For " << nRH << " input PFRecHits, found nEdges = " << nEdges << "  nFracs = " << nFracs
              << "  (GPU) pfClustersFromCuda->size() = " << pfClustersFromCuda->size() << std::endl
              << std::endl;

  numEvents++;
}

void PFClusterProducerCudaHCAL::produce(edm::Event& event, const edm::EventSetup& setup) {
  if (_prodInitClusters)
    event.put(std::move(pfClustersFromCuda), "initialClusters");
  event.put(std::move(pfClustersFromCuda));
}
