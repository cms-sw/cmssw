#include "RecoEcal/EgammaClusterAlgos/interface/PFECALSuperClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClustersGraph.h"
#include "CommonTools/ParticleFlow/interface/PFClusterWidthAlgo.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "Math/GenVector/VectorUtil.h"
#include "TVector2.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>

#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace std;
using namespace std::placeholders;  // for _1, _2, _3...

namespace {
  typedef edm::View<reco::PFCluster> PFClusterView;
  typedef edm::Ptr<reco::PFCluster> PFClusterPtr;
  typedef edm::PtrVector<reco::PFCluster> PFClusterPtrVector;
  typedef std::pair<reco::CaloClusterPtr::key_type, reco::CaloClusterPtr> EEPSPair;

  bool sortByKey(const EEPSPair& a, const EEPSPair& b) { return a.first < b.first; }

  inline double ptFast(const double energy, const math::XYZPoint& position, const math::XYZPoint& origin) {
    const auto v = position - origin;
    return energy * std::sqrt(v.perp2() / v.mag2());
  }

  bool greaterByEt(const CalibratedClusterPtr& x, const CalibratedClusterPtr& y) {
    const math::XYZPoint zero(0, 0, 0);
    const double xpt = ptFast(x->energy(), x->ptr()->position(), zero);
    const double ypt = ptFast(y->energy(), y->ptr()->position(), zero);
    return xpt > ypt;
  }

  bool isSeed(const CalibratedClusterPtr& x, double threshold, bool useETcut) {
    const math::XYZPoint zero(0, 0, 0);
    double e_or_et = x->energy();
    if (useETcut)
      e_or_et = ptFast(e_or_et, x->ptr()->position(), zero);
    return e_or_et > threshold;
  }

  bool isLinkedByRecHit(const CalibratedClusterPtr& x,
                        const CalibratedClusterPtr& seed,
                        const double threshold,
                        const double majority,
                        const double maxDEta,
                        const double maxDPhi) {
    if (seed->energy_nocalib() < threshold) {
      return false;
    }
    const double dEta = std::abs(seed->eta() - x->eta());
    const double dPhi = std::abs(TVector2::Phi_mpi_pi(seed->phi() - x->phi()));
    if (maxDEta < dEta || maxDPhi < dPhi) {
      return false;
    }
    // now see if the clusters overlap in rechits
    const auto& seedHitsAndFractions = seed->ptr()->hitsAndFractions();
    const auto& xHitsAndFractions = x->ptr()->hitsAndFractions();
    double x_rechits_tot = xHitsAndFractions.size();
    double x_rechits_match = 0.0;
    for (const std::pair<DetId, float>& seedHit : seedHitsAndFractions) {
      for (const std::pair<DetId, float>& xHit : xHitsAndFractions) {
        if (seedHit.first == xHit.first) {
          x_rechits_match += 1.0;
        }
      }
    }
    return x_rechits_match / x_rechits_tot > majority;
  }

  bool isClustered(const CalibratedClusterPtr& x,
                   const CalibratedClusterPtr seed,
                   const PFECALSuperClusterAlgo::clustering_type type,
                   const EcalMustacheSCParameters* mustache_params,
                   const EcalSCDynamicDPhiParameters* dynamic_dphi_params,
                   const bool dyn_dphi,
                   const double etawidthSuperCluster,
                   const double phiwidthSuperCluster) {
    const double dphi = std::abs(TVector2::Phi_mpi_pi(seed->phi() - x->phi()));
    const bool passes_dphi =
        ((!dyn_dphi && dphi < phiwidthSuperCluster) ||
         (dyn_dphi && reco::MustacheKernel::inDynamicDPhiWindow(
                          dynamic_dphi_params, seed->eta(), seed->phi(), x->energy_nocalib(), x->eta(), x->phi())));

    if (type == PFECALSuperClusterAlgo::kBOX) {
      return (std::abs(seed->eta() - x->eta()) < etawidthSuperCluster && passes_dphi);
    }
    if (type == PFECALSuperClusterAlgo::kMustache) {
      return (passes_dphi && reco::MustacheKernel::inMustache(
                                 mustache_params, seed->eta(), seed->phi(), x->energy_nocalib(), x->eta(), x->phi()));
    }
    return false;
  }

}  // namespace

PFECALSuperClusterAlgo::PFECALSuperClusterAlgo(const reco::SCProducerCache* cache)
    : beamSpot_(nullptr), SCProducerCache_(cache) {}

void PFECALSuperClusterAlgo::setPFClusterCalibration(const std::shared_ptr<PFEnergyCalibration>& calib) {
  _pfEnergyCalibration = calib;
}

void PFECALSuperClusterAlgo::setTokens(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& cc) {
  inputTagPFClusters_ = cc.consumes<edm::View<reco::PFCluster>>(iConfig.getParameter<edm::InputTag>("PFClusters"));
  inputTagPFClustersES_ =
      cc.consumes<reco::PFCluster::EEtoPSAssociation>(iConfig.getParameter<edm::InputTag>("ESAssociation"));
  inputTagBeamSpot_ = cc.consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("BeamSpot"));

  esEEInterCalibToken_ =
      cc.esConsumes<ESEEIntercalibConstants, ESEEIntercalibConstantsRcd, edm::Transition::BeginLuminosityBlock>();
  esChannelStatusToken_ = cc.esConsumes<ESChannelStatus, ESChannelStatusRcd, edm::Transition::BeginLuminosityBlock>();

  if (_clustype == PFECALSuperClusterAlgo::kMustache) {
    ecalMustacheSCParametersToken_ = cc.esConsumes<EcalMustacheSCParameters, EcalMustacheSCParametersRcd>();
  }
  if (useDynamicDPhi_) {
    ecalSCDynamicDPhiParametersToken_ = cc.esConsumes<EcalSCDynamicDPhiParameters, EcalSCDynamicDPhiParametersRcd>();
  }

  if (useRegression_) {
    const edm::ParameterSet& regconf = iConfig.getParameter<edm::ParameterSet>("regressionConfig");

    regr_ = std::make_unique<SCEnergyCorrectorSemiParm>();
    regr_->setTokens(regconf, cc);
  }

  if (isOOTCollection_ || _clustype == PFECALSuperClusterAlgo::kDeepSC) {  // OOT photons and  DeepSC uses rechits
    inputTagBarrelRecHits_ = cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("barrelRecHits"));
    inputTagEndcapRecHits_ = cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("endcapRecHits"));
  }
  if (_clustype == PFECALSuperClusterAlgo::kDeepSC) {  // DeepSC uses geometry also
    caloTopologyToken_ = cc.esConsumes<CaloTopology, CaloTopologyRecord, edm::Transition::BeginLuminosityBlock>();
    caloGeometryToken_ = cc.esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginLuminosityBlock>();
  }
}

void PFECALSuperClusterAlgo::update(const edm::EventSetup& setup) {
  if (useRegression_) {
    regr_->setEventSetup(setup);
  }

  edm::ESHandle<ESEEIntercalibConstants> esEEInterCalibHandle_ = setup.getHandle(esEEInterCalibToken_);
  _pfEnergyCalibration->initAlphaGamma_ESplanes_fromDB(esEEInterCalibHandle_.product());

  edm::ESHandle<ESChannelStatus> esChannelStatusHandle_ = setup.getHandle(esChannelStatusToken_);
  channelStatus_ = esChannelStatusHandle_.product();

  if (_clustype == PFECALSuperClusterAlgo::kDeepSC) {  // DeepSC uses geometry
    edm::ESHandle<CaloGeometry> caloGeometryHandle_ = setup.getHandle(caloGeometryToken_);
    geometry_ = caloGeometryHandle_.product();
    ebGeom_ = caloGeometryHandle_->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    eeGeom_ = caloGeometryHandle_->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    esGeom_ = caloGeometryHandle_->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

    edm::ESHandle<CaloTopology> caloTopologyHandle_ = setup.getHandle(caloTopologyToken_);
    topology_ = caloTopologyHandle_.product();
  }
}

void PFECALSuperClusterAlgo::updateSCParams(const edm::EventSetup& setup) {
  if (_clustype == PFECALSuperClusterAlgo::kMustache) {
    mustacheSCParams_ = &setup.getData(ecalMustacheSCParametersToken_);
  }
  if (useDynamicDPhi_) {
    scDynamicDPhiParams_ = &setup.getData(ecalSCDynamicDPhiParametersToken_);
  }
}

void PFECALSuperClusterAlgo::loadAndSortPFClusters(const edm::Event& iEvent) {
  //load input collections
  //Load the pfcluster collections
  edm::Handle<edm::View<reco::PFCluster>> pfclustersHandle;
  iEvent.getByToken(inputTagPFClusters_, pfclustersHandle);

  edm::Handle<reco::PFCluster::EEtoPSAssociation> psAssociationHandle;
  iEvent.getByToken(inputTagPFClustersES_, psAssociationHandle);

  const PFClusterView& clusters = *pfclustersHandle.product();
  const reco::PFCluster::EEtoPSAssociation& psclusters = *psAssociationHandle.product();

  //load BeamSpot
  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByToken(inputTagBeamSpot_, bsHandle);
  beamSpot_ = bsHandle.product();

  //initialize regression for this event
  if (useRegression_) {
    regr_->setEvent(iEvent);
  }

  // reset the system for running
  superClustersEB_ = std::make_unique<reco::SuperClusterCollection>();
  _clustersEB.clear();
  superClustersEE_ = std::make_unique<reco::SuperClusterCollection>();
  _clustersEE.clear();
  EEtoPS_ = &psclusters;

  //Select PF clusters available for the clustering
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cluster = clusters.ptrAt(i);
    LogDebug("PFClustering") << "Loading PFCluster i=" << cluster.key() << " energy=" << cluster->energy();

    // protection for sim clusters
    if (cluster->caloID().detectors() == 0 && cluster->hitsAndFractions().empty())
      continue;

    CalibratedClusterPtr calib_cluster(new CalibratedPFCluster(cluster));
    switch (cluster->layer()) {
      case PFLayer::ECAL_BARREL:
        if (calib_cluster->energy() > threshPFClusterBarrel_) {
          _clustersEB.push_back(calib_cluster);
        }
        break;
      case PFLayer::HGCAL:
      case PFLayer::ECAL_ENDCAP:
        if (calib_cluster->energy() > threshPFClusterEndcap_) {
          _clustersEE.push_back(calib_cluster);
        }
        break;
      default:
        break;
    }
  }
  // sort full cluster collections by their calibrated energy
  // this will put all the seeds first by construction
  std::sort(_clustersEB.begin(), _clustersEB.end(), greaterByEt);
  std::sort(_clustersEE.begin(), _clustersEE.end(), greaterByEt);

  // set recHit collections for OOT photons and DeepSC
  if (isOOTCollection_ || _clustype == PFECALSuperClusterAlgo::kDeepSC) {
    edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
    iEvent.getByToken(inputTagBarrelRecHits_, barrelRecHitsHandle);
    if (!barrelRecHitsHandle.isValid()) {
      throw cms::Exception("PFECALSuperClusterAlgo")
          << "If you use OOT photons or DeepSC, need to specify proper barrel rec hit collection";
    }
    barrelRecHits_ = barrelRecHitsHandle.product();

    edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
    iEvent.getByToken(inputTagEndcapRecHits_, endcapRecHitsHandle);
    if (!endcapRecHitsHandle.isValid()) {
      throw cms::Exception("PFECALSuperClusterAlgo")
          << "If you use OOT photons or DeepSC, need to specify proper endcap rec hit collection";
    }
    endcapRecHits_ = endcapRecHitsHandle.product();
  }
}

void PFECALSuperClusterAlgo::run() {
  // clusterize the EB
  buildAllSuperClusters(_clustersEB, threshPFClusterSeedBarrel_);
  // clusterize the EE
  buildAllSuperClusters(_clustersEE, threshPFClusterSeedEndcap_);
}

void PFECALSuperClusterAlgo::buildAllSuperClusters(CalibratedClusterPtrVector& clusters, double seedthresh) {
  if (_clustype == PFECALSuperClusterAlgo::kMustache || _clustype == PFECALSuperClusterAlgo::kBOX)
    buildAllSuperClustersMustacheOrBox(clusters, seedthresh);
  else if (_clustype == PFECALSuperClusterAlgo::kDeepSC)
    buildAllSuperClustersDeepSC(clusters, seedthresh);
}

void PFECALSuperClusterAlgo::buildAllSuperClustersMustacheOrBox(CalibratedClusterPtrVector& clusters,
                                                                double seedthresh) {
  auto seedable = std::bind(isSeed, _1, seedthresh, threshIsET_);

  // make sure only seeds appear at the front of the list of clusters
  std::stable_partition(clusters.begin(), clusters.end(), seedable);

  // in each iteration we are working on a list that is already sorted
  // in the cluster energy and remains so through each iteration
  // NB: since clusters is sorted in loadClusters any_of has O(1)
  //     timing until you run out of seeds!
  while (std::any_of(clusters.cbegin(), clusters.cend(), seedable)) {
    buildSuperClusterMustacheOrBox(clusters.front(), clusters);
  }
}

void PFECALSuperClusterAlgo::buildAllSuperClustersDeepSC(CalibratedClusterPtrVector& clusters, double seedthresh) {
  auto seedable = std::bind(isSeed, _1, seedthresh, threshIsET_);
  // EcalClustersGraph utility class for DeepSC algorithm application
  // make sure only seeds appear at the front of the list of clusters
  auto last_seed = std::stable_partition(clusters.begin(), clusters.end(), seedable);

  reco::EcalClustersGraph ecalClusterGraph_{clusters,
                                            static_cast<int>(std::distance(clusters.begin(), last_seed)),
                                            topology_,
                                            ebGeom_,
                                            eeGeom_,
                                            barrelRecHits_,
                                            endcapRecHits_,
                                            SCProducerCache_};
  // Build sub-regions of the detector where the DeepSC algo will be run
  ecalClusterGraph_.initWindows();
  // For each sub-region, prepare the DeepSC input tensors
  ecalClusterGraph_.fillVariables();
  // Evaluate the DeepSC algorithm and save the scores
  ecalClusterGraph_.evaluateScores();
  // Select the final SuperCluster using the CollectionStrategy defined in the cfi
  ecalClusterGraph_.setThresholds();
  ecalClusterGraph_.selectClusters();
  // Extract the final SuperCluster collection
  reco::EcalClustersGraph::EcalGraphOutput windows = ecalClusterGraph_.getGraphOutput();
  for (auto& [seed, clustered] : windows) {
    bool isEE = false;
    if (seed->ptr()->layer() == PFLayer::ECAL_ENDCAP)
      isEE = true;
    finalizeSuperCluster(seed, clustered, isEE);
  }
}

void PFECALSuperClusterAlgo::buildSuperClusterMustacheOrBox(CalibratedClusterPtr& seed,
                                                            CalibratedClusterPtrVector& clusters) {
  CalibratedClusterPtrVector clustered;

  double etawidthSuperCluster = 0.0;
  double phiwidthSuperCluster = 0.0;
  bool isEE = false;
  switch (seed->ptr()->layer()) {
    case PFLayer::ECAL_BARREL:
      phiwidthSuperCluster = phiwidthSuperClusterBarrel_;
      etawidthSuperCluster = etawidthSuperClusterBarrel_;
      edm::LogInfo("PFClustering") << "Building SC number " << superClustersEB_->size() + 1 << " in the ECAL barrel!";
      break;
    case PFLayer::HGCAL:
    case PFLayer::ECAL_ENDCAP:

      phiwidthSuperCluster = phiwidthSuperClusterEndcap_;
      etawidthSuperCluster = etawidthSuperClusterEndcap_;
      edm::LogInfo("PFClustering") << "Building SC number " << superClustersEE_->size() + 1 << " in the ECAL endcap!"
                                   << std::endl;
      isEE = true;
      break;
    default:
      break;
  }

  auto isClusteredWithSeed = std::bind(isClustered,
                                       _1,
                                       seed,
                                       _clustype,
                                       mustacheSCParams_,
                                       scDynamicDPhiParams_,
                                       useDynamicDPhi_,
                                       etawidthSuperCluster,
                                       phiwidthSuperCluster);

  auto matchesSeedByRecHit = std::bind(isLinkedByRecHit, _1, seed, satelliteThreshold_, fractionForMajority_, 0.1, 0.2);

  // this function shuffles the list of clusters into a list
  // where all clustered sub-clusters are at the front
  // and returns a pointer to the first unclustered cluster.
  // The relative ordering of clusters is preserved
  // (i.e. both resulting sub-lists are sorted by energy).
  auto not_clustered = std::stable_partition(clusters.begin(), clusters.end(), isClusteredWithSeed);
  // satellite cluster merging
  // it was found that large clusters can split!
  if (doSatelliteClusterMerge_) {
    not_clustered = std::stable_partition(not_clustered, clusters.end(), matchesSeedByRecHit);
  }

  if (verbose_) {
    edm::LogInfo("PFClustering") << "Dumping cluster detail";
    edm::LogVerbatim("PFClustering") << "\tPassed seed: e = " << seed->energy_nocalib() << " eta = " << seed->eta()
                                     << " phi = " << seed->phi() << std::endl;
    for (auto clus = clusters.cbegin(); clus != not_clustered; ++clus) {
      edm::LogVerbatim("PFClustering") << "\t\tClustered cluster: e = " << (*clus)->energy_nocalib()
                                       << " eta = " << (*clus)->eta() << " phi = " << (*clus)->phi() << std::endl;
    }
    for (auto clus = not_clustered; clus != clusters.end(); ++clus) {
      edm::LogVerbatim("PFClustering") << "\tNon-Clustered cluster: e = " << (*clus)->energy_nocalib()
                                       << " eta = " << (*clus)->eta() << " phi = " << (*clus)->phi() << std::endl;
    }
  }

  if (not_clustered == clusters.begin()) {
    if (dropUnseedable_) {
      clusters.erase(clusters.begin());
      return;
    } else {
      throw cms::Exception("PFECALSuperClusterAlgo::buildSuperCluster")
          << "Cluster is not seedable!" << std::endl
          << "\tNon-Clustered cluster: e = " << (*not_clustered)->energy_nocalib()
          << " eta = " << (*not_clustered)->eta() << " phi = " << (*not_clustered)->phi() << std::endl;
    }
  }

  // move the clustered clusters out of available cluster list
  // and into a temporary vector for building the SC
  CalibratedClusterPtrVector clustered_tmp(clusters.begin(), not_clustered);
  clustered = clustered_tmp;
  clusters.erase(clusters.begin(), not_clustered);

  // Finalize the SuperCluster passing the list of clustered clusters
  finalizeSuperCluster(seed, clustered, isEE);
}

void PFECALSuperClusterAlgo::finalizeSuperCluster(CalibratedClusterPtr& seed,
                                                  CalibratedClusterPtrVector& clustered,
                                                  bool isEE) {
  // need the vector of raw pointers for a PF width class
  std::vector<const reco::PFCluster*> bare_ptrs;
  // calculate necessary parameters and build the SC
  double posX(0), posY(0), posZ(0), corrSCEnergy(0), corrPS1Energy(0), corrPS2Energy(0), energyweight(0),
      energyweighttot(0);
  for (const auto& clus : clustered) {
    double ePS1 = 0.0;
    double ePS2 = 0.0;
    energyweight = clus->energy_nocalib();
    bare_ptrs.push_back(clus->ptr().get());
    // update EE calibrated super cluster energies
    if (isEE) {
      auto ee_key_val = std::make_pair(clus->ptr().key(), edm::Ptr<reco::PFCluster>());
      const auto clustops = std::equal_range(EEtoPS_->begin(), EEtoPS_->end(), ee_key_val, sortByKey);
      std::vector<reco::PFCluster const*> psClusterPointers;
      for (auto i_ps = clustops.first; i_ps != clustops.second; ++i_ps) {
        psClusterPointers.push_back(i_ps->second.get());
      }
      auto calibratedEnergies = _pfEnergyCalibration->calibrateEndcapClusterEnergies(
          *(clus->ptr()), psClusterPointers, *channelStatus_, applyCrackCorrections_);
      ePS1 = calibratedEnergies.ps1Energy;
      ePS2 = calibratedEnergies.ps2Energy;
    }

    if (ePS1 == -1.)
      ePS1 = 0;
    if (ePS2 == -1.)
      ePS2 = 0;

    switch (_eweight) {
      case kRaw:  // energyweight is initialized to raw cluster energy
        break;
      case kCalibratedNoPS:
        energyweight = clus->energy() - ePS1 - ePS2;
        break;
      case kCalibratedTotal:
        energyweight = clus->energy();
        break;
      default:
        break;
    }
    const math::XYZPoint& cluspos = clus->ptr()->position();
    posX += energyweight * cluspos.X();
    posY += energyweight * cluspos.Y();
    posZ += energyweight * cluspos.Z();

    energyweighttot += energyweight;
    corrSCEnergy += clus->energy();
    corrPS1Energy += ePS1;
    corrPS2Energy += ePS2;
  }
  posX /= energyweighttot;
  posY /= energyweighttot;
  posZ /= energyweighttot;

  // now build the supercluster
  reco::SuperCluster new_sc(corrSCEnergy, math::XYZPoint(posX, posY, posZ));
  new_sc.setCorrectedEnergy(corrSCEnergy);
  new_sc.setSeed(clustered.front()->ptr());
  new_sc.setPreshowerEnergy(corrPS1Energy + corrPS2Energy);
  new_sc.setPreshowerEnergyPlane1(corrPS1Energy);
  new_sc.setPreshowerEnergyPlane2(corrPS2Energy);
  for (const auto& clus : clustered) {
    new_sc.addCluster(clus->ptr());

    auto& hits_and_fractions = clus->ptr()->hitsAndFractions();
    for (auto& hit_and_fraction : hits_and_fractions) {
      new_sc.addHitAndFraction(hit_and_fraction.first, hit_and_fraction.second);
    }
    if (isEE) {
      auto ee_key_val = std::make_pair(clus->ptr().key(), edm::Ptr<reco::PFCluster>());
      const auto clustops = std::equal_range(EEtoPS_->begin(), EEtoPS_->end(), ee_key_val, sortByKey);
      // EE rechits should be uniquely matched to sets of pre-shower
      // clusters at this point, so we throw an exception if otherwise
      // now wrapped in EDM debug flags
      for (auto i_ps = clustops.first; i_ps != clustops.second; ++i_ps) {
        edm::Ptr<reco::PFCluster> psclus(i_ps->second);
#ifdef EDM_ML_DEBUG

        auto found_pscluster =
            std::find(new_sc.preshowerClustersBegin(), new_sc.preshowerClustersEnd(), reco::CaloClusterPtr(psclus));

        if (found_pscluster == new_sc.preshowerClustersEnd()) {
#endif
          new_sc.addPreshowerCluster(psclus);
#ifdef EDM_ML_DEBUG
        } else {
          throw cms::Exception("PFECALSuperClusterAlgo::buildSuperCluster")
              << "Found a PS cluster matched to more than one EE cluster!" << std::endl
              << std::hex << psclus.get() << " == " << found_pscluster->get() << std::dec << std::endl;
        }
#endif
      }
    }
  }

  // calculate linearly weighted cluster widths
  PFClusterWidthAlgo pfwidth(bare_ptrs);
  new_sc.setEtaWidth(pfwidth.pflowEtaWidth());
  new_sc.setPhiWidth(pfwidth.pflowPhiWidth());

  // cache the value of the raw energy
  new_sc.rawEnergy();

  //apply regression energy corrections
  if (useRegression_) {
    regr_->modifyObject(new_sc);
  }

  // save the super cluster to the appropriate list (if it passes the final
  // Et threshold)
  //Note that Et is computed here with respect to the beamspot position
  //in order to be consistent with the cut applied in the
  //ElectronSeedProducer
  double scEtBS = ptFast(new_sc.energy(), new_sc.position(), beamSpot_->position());

  if (scEtBS > threshSuperClusterEt_) {
    switch (seed->ptr()->layer()) {
      case PFLayer::ECAL_BARREL:
        if (isOOTCollection_) {
          DetId seedId = new_sc.seed()->seed();
          EcalRecHitCollection::const_iterator seedRecHit = barrelRecHits_->find(seedId);
          if (!seedRecHit->checkFlag(EcalRecHit::kOutOfTime))
            break;
        }
        superClustersEB_->push_back(new_sc);
        break;
      case PFLayer::HGCAL:
      case PFLayer::ECAL_ENDCAP:
        if (isOOTCollection_) {
          DetId seedId = new_sc.seed()->seed();
          EcalRecHitCollection::const_iterator seedRecHit = endcapRecHits_->find(seedId);
          if (!seedRecHit->checkFlag(EcalRecHit::kOutOfTime))
            break;
        }
        superClustersEE_->push_back(new_sc);
        break;
      default:
        break;
    }
  }
}
