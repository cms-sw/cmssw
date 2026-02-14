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

    // Reusable buffers to avoid per-event allocations/copies.
  static constexpr unsigned int kRegressionFeatureCount = 8;

  cms::Ort::FloatArrays onnxInputs_;
  cms::Ort::FloatArrays onnxOutputs_;
  std::vector<std::vector<int64_t>> onnxInputShapes_;
  std::vector<float> regressionInputs_;

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


  if (enableRegression_) {
    onnxInputs_.resize(1);
    // rank-2: [batch, features]
    onnxInputShapes_.assign(1, std::vector<int64_t>(2, 0));
    // Reserve a reasonable default; will be grown if needed.
    regressionInputs_.reserve(1024u * kRegressionFeatureCount);
  }
  produces<reco::SuperClusterCollection>();
  produces<reco::CaloClusterCollection>();  // The CaloCluster corresponding to each EM trackster
}

void EGammaSuperclusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if (enableRegression_ && UNLIKELY(globalCache() == nullptr)) {
    throw cms::Exception("Configuration")
        << "EGammaSuperclusterProducer: enableRegression=true but GlobalCache<ONNXRuntime> is null.";
  }
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
  regressionInputs_.clear();
  regressionInputs_.reserve(ticlSuperclusters.size() * kRegressionFeatureCount);

  unsigned int superclustersPassingSelectionsCount = 0;

  for (std::size_t sc_i = 0; sc_i < ticlSuperclusters.size(); ++sc_i) {
    auto const& ticlSupercluster = ticlSuperclusters[sc_i];
    if (ticlSupercluster.raw_pt() < superclusterEtThreshold_) {
      continue;
    }

    auto const& superclusterLink = ticlSuperclusterLinks[sc_i];
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
    ++superclustersPassingSelectionsCount;

    if (enableRegression_) {
      const auto& seedTs = emTracksters[superclusterLink[0]];
      const float lastE = (superclusterLink.size() >= 2) ? emTracksters[superclusterLink.back()].raw_energy() : 0.f;

      const float etaWidth = std::abs(max_eta - min_eta);
      const float phiWidth = std::abs(deltaPhi(max_phi, min_phi));

      const float seedMinusLastE = seedTs.raw_energy() - lastE;
      const float seedFrac = seedTs.raw_energy() / ticlSupercluster.raw_energy();
      const float nTs = static_cast<float>(superclusterLink.size());

      const size_t base = regressionInputs_.size();
      regressionInputs_.resize(base + kRegressionFeatureCount);

      regressionInputs_[base + 0] = ticlSupercluster.barycenter().eta();
      regressionInputs_[base + 1] = ticlSupercluster.barycenter().phi();
      regressionInputs_[base + 2] = ticlSupercluster.raw_energy();
      regressionInputs_[base + 3] = etaWidth;
      regressionInputs_[base + 4] = phiWidth;
      regressionInputs_[base + 5] = seedMinusLastE;
      regressionInputs_[base + 6] = seedFrac;
      regressionInputs_[base + 7] = nTs;
    }
  
  }

  if (enableRegression_ && superclustersPassingSelectionsCount > 0u) {
    // shape: [batch, features]
    onnxInputShapes_[0][0] = static_cast<int64_t>(superclustersPassingSelectionsCount);
    onnxInputShapes_[0][1] = static_cast<int64_t>(kRegressionFeatureCount);

    // Hand buffer ownership to ONNXRuntime without realloc/copy.
    onnxInputs_[0].swap(regressionInputs_);

    onnxOutputs_.clear();
    static const std::vector<std::string> kInputNames = {"input"};
    globalCache()->runInto(kInputNames,
                           onnxInputs_,
                           onnxInputShapes_,
                           {},               // all outputs
                           onnxOutputs_,      // resized as needed
                           {},               // optional output shapes
                           static_cast<int64_t>(superclustersPassingSelectionsCount));

    if (onnxOutputs_.empty()) {
      throw cms::Exception("RuntimeError") << "Regression model returned no outputs.";
    }

    auto const& out = onnxOutputs_[0];
    if (out.size() < egammaSuperclusters->size()) {
      throw cms::Exception("RuntimeError")
          << "Regression output size " << out.size()
          << " smaller than number of produced superclusters " << egammaSuperclusters->size();
    }

    for (std::size_t i = 0; i < egammaSuperclusters->size(); ++i) {
      (*egammaSuperclusters)[i].setCorrectedEnergy(out[i]);
    }

    // Restore buffer for reuse.
    regressionInputs_.swap(onnxInputs_[0]);
    regressionInputs_.clear();
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
