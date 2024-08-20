// Authors : Theo Cuisset <theo.cuisset@cern.ch>, Shamik Ghosh <shamik.ghosh@cern.ch>
// Date : 01/2024
/* 
Translates TICL superclusters to ECAL supercluster dataformats (reco::SuperCluster and reco::CaloCluster).
Performs similar task as RecoEcal/EgammaCLusterProducers/PFECALSuperClusterProducer
Note that all tracksters are translated to reco::CaloCluster, even those that are not in any SuperCluster
*/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/FileInPath.h"

#include <vector>
#include <array>
#include <limits>
#include <algorithm>

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/deltaPhi.h"

using cms::Ort::ONNXRuntime;

class EGammaSuperclusterProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  EGammaSuperclusterProducer(const edm::ParameterSet&, const ONNXRuntime*);

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet& iConfig);
  static void globalEndJob(const ONNXRuntime*);

private:
  edm::EDGetTokenT<ticl::TracksterCollection> ticlSuperClustersToken_;
  edm::EDGetTokenT<std::vector<std::vector<unsigned int>>> superClusterLinksToken_;
  edm::EDGetTokenT<ticl::TracksterCollection> ticlTrackstersEMToken_;
  edm::EDGetTokenT<reco::CaloClusterCollection> layerClustersToken_;
  float superclusterEtThreshold_;
  bool enableRegression_;
};

EGammaSuperclusterProducer::EGammaSuperclusterProducer(const edm::ParameterSet& ps, const ONNXRuntime*)
    : ticlSuperClustersToken_(consumes<ticl::TracksterCollection>(ps.getParameter<edm::InputTag>("ticlSuperClusters"))),
      superClusterLinksToken_(consumes<std::vector<std::vector<unsigned int>>>(
          edm::InputTag(ps.getParameter<edm::InputTag>("ticlSuperClusters").label(),
                        "linkedTracksterIdToInputTracksterId",
                        ps.getParameter<edm::InputTag>("ticlSuperClusters").process()))),
      ticlTrackstersEMToken_(consumes<ticl::TracksterCollection>(ps.getParameter<edm::InputTag>("ticlTrackstersEM"))),
      layerClustersToken_(consumes<reco::CaloClusterCollection>(ps.getParameter<edm::InputTag>("layerClusters"))),
      superclusterEtThreshold_(ps.getParameter<double>("superclusterEtThreshold")),
      enableRegression_(ps.getParameter<bool>("enableRegression")) {
  produces<reco::SuperClusterCollection>();
  produces<reco::CaloClusterCollection>();  // The CaloCluster corresponding to each EM trackster
}

void EGammaSuperclusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& ticlSuperclusters = iEvent.get(ticlSuperClustersToken_);
  auto const& ticlSuperclusterLinks = iEvent.get(superClusterLinksToken_);
  auto emTracksters_h = iEvent.getHandle(ticlTrackstersEMToken_);
  auto const& emTracksters = *emTracksters_h;
  auto const& layerClusters = iEvent.get(layerClustersToken_);
  // Output collections :
  auto egammaSuperclusters = std::make_unique<reco::SuperClusterCollection>();
  auto caloClustersEM = std::make_unique<reco::CaloClusterCollection>();

  // Fill reco::CaloCluster collection (1-1 mapping to TICL EM tracksters)
  for (ticl::Trackster const& emTrackster : emTracksters) {
    std::vector<std::pair<DetId, float>> hitsAndFractions;
    int iLC = 0;
    std::for_each(std::begin(emTrackster.vertices()), std::end(emTrackster.vertices()), [&](unsigned int lcId) {
      const auto fraction = 1.f / emTrackster.vertex_multiplicity(iLC++);
      for (const auto& cell : layerClusters[lcId].hitsAndFractions()) {
        hitsAndFractions.emplace_back(cell.first, cell.second * fraction);
      }
    });

    reco::CaloCluster& caloCluster = caloClustersEM->emplace_back(
        emTrackster.raw_energy(),                  // energy
        math::XYZPoint(emTrackster.barycenter()),  // position
        reco::CaloID(reco::CaloID::DET_HGCAL_ENDCAP),
        hitsAndFractions,
        reco::CaloCluster::particleFlow,  // algoID (copying from output of PFECALCSuperClusterProducer)
        hitsAndFractions.at(0)
            .first  // seedId (this may need to be updated once a common definition of the seed of a cluster is adopted for Phase-2)
    );
    caloCluster.setCorrectedEnergy(emTrackster.raw_energy());  // Needs to be updated with new supercluster regression
  }

  edm::OrphanHandle<reco::CaloClusterCollection> caloClustersEM_h = iEvent.put(std::move(caloClustersEM));

  // Fill reco::SuperCluster collection and prepare regression inputs
  assert(ticlSuperclusters.size() == ticlSuperclusterLinks.size());
  const unsigned int regressionFeatureCount = 8;
  std::vector<float> regressionInputs;
  regressionInputs.reserve(ticlSuperclusters.size() * regressionFeatureCount);
  unsigned int superclustersPassingSelectionsCount = 0;
  for (std::size_t sc_i = 0; sc_i < ticlSuperclusters.size(); sc_i++) {
    ticl::Trackster const& ticlSupercluster = ticlSuperclusters[sc_i];
    if (ticlSupercluster.raw_pt() < superclusterEtThreshold_)
      continue;
    std::vector<unsigned int> const& superclusterLink = ticlSuperclusterLinks[sc_i];
    assert(!superclusterLink.empty());

    reco::CaloClusterPtrVector trackstersEMInSupercluster;
    float max_eta = std::numeric_limits<float>::min();
    float max_phi = std::numeric_limits<float>::min();
    float min_eta = std::numeric_limits<float>::max();
    float min_phi = std::numeric_limits<float>::max();
    for (unsigned int tsInSc_id : superclusterLink) {
      trackstersEMInSupercluster.push_back(reco::CaloClusterPtr(caloClustersEM_h, tsInSc_id));
      ticl::Trackster const& constituentTrackster = emTracksters[tsInSc_id];
      max_eta = std::max(constituentTrackster.barycenter().eta(), max_eta);
      max_phi = std::max(constituentTrackster.barycenter().phi(), max_phi);
      min_eta = std::min(constituentTrackster.barycenter().eta(), min_eta);
      min_phi = std::min(constituentTrackster.barycenter().phi(), min_phi);
    }
    egammaSuperclusters->emplace_back(
        ticlSupercluster.raw_energy(),
        reco::SuperCluster::Point(ticlSupercluster.barycenter()),
        reco::CaloClusterPtr(caloClustersEM_h,
                             superclusterLink[0]),  // seed (first trackster in superclusterLink is the seed)
        trackstersEMInSupercluster,                 // clusters
        0.,                                         // Epreshower (not relevant for HGCAL)
        -1,                                         // phiwidth (not implemented yet)
        -1                                          // etawidth (not implemented yet)
    );
    superclustersPassingSelectionsCount++;

    if (enableRegression_) {
      regressionInputs.insert(
          regressionInputs.end(),
          {ticlSupercluster.barycenter().eta(),
           ticlSupercluster.barycenter().phi(),
           ticlSupercluster.raw_energy(),
           std::abs(max_eta - min_eta),
           max_phi - min_phi > M_PI ? 2 * static_cast<float>(M_PI) - (max_phi - min_phi) : max_phi - min_phi,
           emTracksters[superclusterLink[0]].raw_energy() -
               (superclusterLink.size() >= 2 ? emTracksters[superclusterLink.back()].raw_energy() : 0.f),
           emTracksters[superclusterLink[0]].raw_energy() / ticlSupercluster.raw_energy(),
           static_cast<float>(superclusterLink.size())});
    }
  }

  if (enableRegression_ && superclustersPassingSelectionsCount > 0) {
    // Run the regression
    // ONNXRuntime takes std::vector<std::vector<float>>& as input (non-const reference) so we have to make a new vector
    std::vector<std::vector<float>> inputs_for_onnx{{std::move(regressionInputs)}};
    std::vector<float> outputs =
        globalCache()->run({"input"}, inputs_for_onnx, {}, {}, superclustersPassingSelectionsCount)[0];

    assert(egammaSuperclusters->size() == outputs.size());
    for (std::size_t sc_i = 0; sc_i < egammaSuperclusters->size(); sc_i++) {
      (*egammaSuperclusters)[sc_i].setCorrectedEnergy(outputs[sc_i]);
      // correctedEnergyUncertainty is left at its default value
    }
  }

  iEvent.put(std::move(egammaSuperclusters));
}

std::unique_ptr<ONNXRuntime> EGammaSuperclusterProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  if (iConfig.getParameter<bool>("enableRegression"))
    return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("regressionModelPath").fullPath());
  else
    return std::unique_ptr<ONNXRuntime>(nullptr);
}

void EGammaSuperclusterProducer::globalEndJob(const ONNXRuntime*) {}

void EGammaSuperclusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ticlSuperClusters", edm::InputTag("ticlTracksterLinksSuperclusteringDNN"));
  desc.add<edm::InputTag>("ticlTrackstersEM", edm::InputTag("ticlTrackstersCLUE3DHigh"))
      ->setComment("The trackster collection used before superclustering, ie CLUE3D EM tracksters");
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"))
      ->setComment("The layer cluster collection that goes with ticlTrackstersEM");
  desc.add<double>("superclusterEtThreshold", 4.)->setComment("Minimum supercluster transverse energy.");
  desc.add<bool>("enableRegression", true)->setComment("Enable supercluster energy regression");
  desc.add<edm::FileInPath>("regressionModelPath",
                            edm::FileInPath("RecoHGCal/TICL/data/superclustering/regression_v1.onnx"))
      ->setComment("Path to regression network (as ONNX model)");

  descriptions.add("ticlEGammaSuperClusterProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EGammaSuperclusterProducer);
