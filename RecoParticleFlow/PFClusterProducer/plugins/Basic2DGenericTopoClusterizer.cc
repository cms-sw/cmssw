#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"

class Basic2DGenericTopoClusterizer : public InitialClusteringStepBase {
  typedef Basic2DGenericTopoClusterizer B2DGT;

public:
  Basic2DGenericTopoClusterizer(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
      : InitialClusteringStepBase(conf, cc), _useCornerCells(conf.getParameter<bool>("useCornerCells")) {}
  ~Basic2DGenericTopoClusterizer() override = default;
  Basic2DGenericTopoClusterizer(const B2DGT&) = delete;
  B2DGT& operator=(const B2DGT&) = delete;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
                     const std::vector<bool>&,
                     const std::vector<bool>&,
                     reco::PFClusterCollection&) override;

private:
  const bool _useCornerCells;
  void buildTopoCluster(const edm::Handle<reco::PFRecHitCollection>&,
                        const std::vector<bool>&,  // masked rechits
                        unsigned int,              //present rechit
                        std::vector<bool>&,        // hit usage state
                        reco::PFCluster&);         // the topocluster
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory, Basic2DGenericTopoClusterizer, "Basic2DGenericTopoClusterizer");

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

void Basic2DGenericTopoClusterizer::buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
                                                  const std::vector<bool>& rechitMask,
                                                  const std::vector<bool>& seedable,
                                                  reco::PFClusterCollection& output) {
  auto const& hits = *input;
  std::vector<bool> used(hits.size(), false);
  std::vector<unsigned int> seeds;

  // get the seeds and sort them descending in energy
  seeds.reserve(hits.size());
  for (unsigned int i = 0; i < hits.size(); ++i) {
    if (!rechitMask[i] || !seedable[i] || used[i])
      continue;
    seeds.emplace_back(i);
  }
  // maxHeap would be better
  std::sort(
      seeds.begin(), seeds.end(), [&](unsigned int i, unsigned int j) { return hits[i].energy() > hits[j].energy(); });

  reco::PFCluster temp;
  for (auto seed : seeds) {
    if (!rechitMask[seed] || !seedable[seed] || used[seed])
      continue;
    temp.reset();
    buildTopoCluster(input, rechitMask, seed, used, temp);
    if (!temp.recHitFractions().empty())
      output.push_back(temp);
  }
}

void Basic2DGenericTopoClusterizer::buildTopoCluster(const edm::Handle<reco::PFRecHitCollection>& input,
                                                     const std::vector<bool>& rechitMask,
                                                     unsigned int kcell,
                                                     std::vector<bool>& used,
                                                     reco::PFCluster& topocluster) {
  auto const& cell = (*input)[kcell];
  int cell_layer = (int)cell.layer();
  if (cell_layer == PFLayer::HCAL_BARREL2 && std::abs(cell.positionREP().eta()) > 0.34) {
    cell_layer *= 100;
  }

  auto const& thresholds = _thresholds.find(cell_layer)->second;
  double thresholdE = 0.;
  double thresholdPT2 = 0.;

  for (unsigned int j = 0; j < (std::get<1>(thresholds)).size(); ++j) {
    int depth = std::get<0>(thresholds)[j];

    if ((cell_layer == PFLayer::HCAL_BARREL1 && cell.depth() == depth) ||
        (cell_layer == PFLayer::HCAL_ENDCAP && cell.depth() == depth) ||
        (cell_layer != PFLayer::HCAL_BARREL1 && cell_layer != PFLayer::HCAL_ENDCAP)) {
      thresholdE = std::get<1>(thresholds)[j];
      thresholdPT2 = std::get<2>(thresholds)[j];
    }
  }

  if (cell.energy() < thresholdE || cell.pt2() < thresholdPT2) {
    LOGDRESSED("GenericTopoCluster::buildTopoCluster()")
        << "RecHit " << cell.detId() << " with enegy " << cell.energy() << " GeV was rejected!." << std::endl;
    return;
  }

  auto k = kcell;
  used[k] = true;
  auto ref = makeRefhit(input, k);
  topocluster.addRecHitFraction(reco::PFRecHitFraction(ref, 1.0));

  auto const& neighbours = (_useCornerCells ? cell.neighbours8() : cell.neighbours4());

  for (auto nb : neighbours) {
    if (used[nb] || !rechitMask[nb]) {
      LOGDRESSED("GenericTopoCluster::buildTopoCluster()")
          << "  RecHit " << cell.detId() << "\'s"
          << " neighbor RecHit " << input->at(nb).detId() << " with enegy " << input->at(nb).energy()
          << " GeV was rejected!"
          << " Reasons : " << used[nb] << " (used) " << !rechitMask[nb] << " (masked)." << std::endl;
      continue;
    }
    buildTopoCluster(input, rechitMask, nb, used, topocluster);
  }
}
