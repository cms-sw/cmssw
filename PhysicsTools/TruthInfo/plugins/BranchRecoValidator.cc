// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Generic reco-side DQM validator: matches any reco collection to the truth Branch
// graph through shared detector hits and books MultiTrackValidator / HGCalValidator
// style efficiency / fake-rate / merge / duplicate plots plus a match-purity
// distribution. The reco object is reduced to a list of truth::RecoHit by the
// truth::recoHits adapters (RecoHitAdapters.h), so adding a new reco type is just a
// new adapter + a Traits policy - no change to the validation logic. Two concrete
// modules are instantiated from the one template:
//   * BranchTrackRecoValidator    - reco::Track, tracker channel, shared-hit
//                                    multiplicity (no per-cell energy in the tracker).
//   * BranchTracksterRecoValidator - ticl::Trackster, calo channel, shared energy.
// The truth side (efficiency denominator) is the set of "interesting" branches
// (interestingPdgIds, empty = all) that carry hits in the relevant channel and pass
// a kinematic selection; the harvester (DQMGenericClient) forms the ratios.

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/RecoHitAdapters.h"

namespace {
  // One reco object reduced to what the matcher needs: its hits + kinematics. The
  // second kinematic axis x is p_T for tracks and (raw) energy for tracksters.
  struct RecoObject {
    std::vector<truth::RecoHit> hits;
    double eta = 0.;
    double x = 0.;
  };

  // --- Traits: per-reco-type EDM access, metric and kinematics policy. ---

  // reco::Track: tracker channel, shared-hit multiplicity, x = p_T.
  class TrackRecoTraits {
  public:
    TrackRecoTraits(edm::ParameterSet const& cfg, edm::ConsumesCollector cc)
        : token_(cc.consumes<edm::View<reco::Track>>(cfg.getParameter<edm::InputTag>("recoCollection"))) {}

    std::vector<RecoObject> objects(edm::Event const& event) const {
      std::vector<RecoObject> out;
      auto const& tracks = event.get(token_);
      out.reserve(tracks.size());
      for (auto const& t : tracks)
        out.push_back(RecoObject{truth::recoHits(t), t.eta(), t.pt()});
      return out;
    }
    static constexpr truth::BranchHitAssociator::Metric metric() {
      return truth::BranchHitAssociator::Metric::SharedHits;
    }
    static constexpr bool useTracker() { return true; }
    static double particleX(math::XYZTLorentzVectorD const& p) { return p.pt(); }
    static void fillDescriptions(edm::ParameterSetDescription& desc) {
      desc.add<edm::InputTag>("recoCollection", edm::InputTag("generalTracks"));
    }

  private:
    edm::EDGetTokenT<edm::View<reco::Track>> token_;
  };

  // ticl::Trackster: calo channel, shared energy (cell fractions), x = raw energy.
  class TracksterRecoTraits {
  public:
    TracksterRecoTraits(edm::ParameterSet const& cfg, edm::ConsumesCollector cc)
        : tracksterToken_(cc.consumes<std::vector<ticl::Trackster>>(cfg.getParameter<edm::InputTag>("recoCollection"))),
          layerClusterToken_(
              cc.consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))) {}

    std::vector<RecoObject> objects(edm::Event const& event) const {
      std::vector<RecoObject> out;
      auto const& tracksters = event.get(tracksterToken_);
      auto const& layerClusters = event.get(layerClusterToken_);
      out.reserve(tracksters.size());
      for (auto const& t : tracksters)
        out.push_back(RecoObject{truth::recoHits(t, layerClusters), t.barycenter().eta(), t.raw_energy()});
      return out;
    }
    static constexpr truth::BranchHitAssociator::Metric metric() {
      return truth::BranchHitAssociator::Metric::SharedEnergy;
    }
    static constexpr bool useTracker() { return false; }
    static double particleX(math::XYZTLorentzVectorD const& p) { return p.energy(); }
    static void fillDescriptions(edm::ParameterSetDescription& desc) {
      desc.add<edm::InputTag>("recoCollection", edm::InputTag("ticlTrackstersCLUE3DHigh"));
      desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
    }

  private:
    edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken_;
    edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClusterToken_;
  };
}  // namespace

template <class Traits>
class BranchRecoValidatorT : public DQMEDAnalyzer {
public:
  explicit BranchRecoValidatorT(edm::ParameterSet const&);
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  [[nodiscard]] bool selected(double eta, double x) const {
    return std::abs(eta) >= minAbsEta_ && std::abs(eta) <= maxAbsEta_ && x >= minX_;
  }
  // Per-object span hits in the relevant channel (calo subgraph or tracker subgraph).
  [[nodiscard]] std::span<const truth::LogicalGraphHitIndex::Hit> channelHits(
      truth::LogicalGraphHitIndex const& hitIndex, uint32_t root) const {
    return Traits::useTracker() ? hitIndex.trackerSubgraphHits(root) : hitIndex.subgraphHits(root);
  }

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const std::vector<int> interestingPdgIds_;
  const std::string folder_;
  const std::string xName_;
  const std::string xTitle_;
  const double xMax_;
  const double minX_;
  const double minAbsEta_;
  const double maxAbsEta_;
  const double matchThreshold_;
  const double mergeThreshold_;
  Traits traits_;

  // Truth (sim) side, vs eta and vs x.
  MonitorElement* denomEta_ = nullptr;
  MonitorElement* denomX_ = nullptr;
  MonitorElement* effNumEta_ = nullptr;
  MonitorElement* effNumX_ = nullptr;
  MonitorElement* dupNumEta_ = nullptr;
  MonitorElement* dupNumX_ = nullptr;
  // Reco side, vs eta and vs x.
  MonitorElement* recoDenomEta_ = nullptr;
  MonitorElement* recoDenomX_ = nullptr;
  MonitorElement* fakeNumEta_ = nullptr;
  MonitorElement* fakeNumX_ = nullptr;
  MonitorElement* mergeNumEta_ = nullptr;
  MonitorElement* mergeNumX_ = nullptr;
  MonitorElement* purity_ = nullptr;
};

template <class Traits>
BranchRecoValidatorT<Traits>::BranchRecoValidatorT(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      interestingPdgIds_(cfg.getParameter<std::vector<int>>("interestingPdgIds")),
      folder_(cfg.getParameter<std::string>("folder")),
      xName_(cfg.getParameter<std::string>("xName")),
      xTitle_(cfg.getParameter<std::string>("xTitle")),
      xMax_(cfg.getParameter<double>("xMax")),
      minX_(cfg.getParameter<double>("minX")),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")),
      matchThreshold_(cfg.getParameter<double>("matchThreshold")),
      mergeThreshold_(cfg.getParameter<double>("mergeThreshold")),
      traits_(cfg, consumesCollector()) {}

template <class Traits>
void BranchRecoValidatorT<Traits>::bookHistograms(DQMStore::IBooker& ib, edm::Run const&, edm::EventSetup const&) {
  ib.setCurrentFolder(folder_);
  constexpr int kEtaBins = 40;
  const double etaMax = maxAbsEta_ + 0.2;
  constexpr int kXBins = 50;
  const char* xt = xTitle_.c_str();

  auto bookEta = [&](char const* n, char const* t) { return ib.book1D(n, t, kEtaBins, -etaMax, etaMax); };
  auto bookX = [&](std::string const& n, std::string const& t) { return ib.book1D(n, t, kXBins, 0., xMax_); };

  denomEta_ = bookEta("denom_eta", "Selected branches vs #eta;#eta;branches");
  denomX_ = bookX("denom_" + xName_, std::string("Selected branches vs ") + xt + ";" + xt + ";branches");
  effNumEta_ = bookEta("effnum_eta", "Branches matched by reco vs #eta;#eta;branches");
  effNumX_ = bookX("effnum_" + xName_, std::string("Branches matched by reco vs ") + xt + ";" + xt + ";branches");
  dupNumEta_ = bookEta("dupnum_eta", "Branches matched by >1 reco vs #eta;#eta;branches");
  dupNumX_ = bookX("dupnum_" + xName_, std::string("Branches matched by >1 reco vs ") + xt + ";" + xt + ";branches");

  recoDenomEta_ = bookEta("recodenom_eta", "Reco objects vs #eta;#eta;reco objects");
  recoDenomX_ = bookX("recodenom_" + xName_, std::string("Reco objects vs ") + xt + ";" + xt + ";reco objects");
  fakeNumEta_ = bookEta("fakenum_eta", "Unmatched (fake) reco vs #eta;#eta;reco objects");
  fakeNumX_ = bookX("fakenum_" + xName_, std::string("Unmatched (fake) reco vs ") + xt + ";" + xt + ";reco objects");
  mergeNumEta_ = bookEta("mergenum_eta", "Reco matched to >1 branch vs #eta;#eta;reco objects");
  mergeNumX_ =
      bookX("mergenum_" + xName_, std::string("Reco matched to >1 branch vs ") + xt + ";" + xt + ";reco objects");

  purity_ = ib.book1D("purity", "Best-branch match purity;purity;reco objects", 52, -0.01, 1.03);
}

template <class Traits>
void BranchRecoValidatorT<Traits>::analyze(edm::Event const& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);

  // Candidate / associator roots = the interesting particles (empty config -> all).
  std::vector<uint32_t> roots;
  if (!interestingPdgIds_.empty()) {
    for (uint32_t i = 0; i < graph.nParticles(); ++i) {
      const int pdgId = graph.particles[i].pdgId;
      if (std::find(interestingPdgIds_.begin(), interestingPdgIds_.end(), pdgId) != interestingPdgIds_.end())
        roots.push_back(i);
    }
  }
  truth::BranchHitAssociator assoc(hitIndex, roots, Traits::metric(), Traits::useTracker());

  // Reco -> sim: match every reco object, accumulate per-branch match multiplicity.
  std::unordered_map<uint32_t, int> recoMatchCount;
  for (auto const& obj : traits_.objects(event)) {
    if (obj.hits.empty() || !selected(obj.eta, obj.x))
      continue;
    recoDenomEta_->Fill(obj.eta);
    recoDenomX_->Fill(obj.x);

    double objWeight = 0.;
    for (auto const& h : obj.hits)
      objWeight += static_cast<double>(h.fraction) * h.energy;
    if (objWeight <= 0.)
      objWeight = 1.;

    auto matches = assoc.bestBranches(std::span<const truth::RecoHit>(obj.hits));
    const double bestPurity = matches.empty() ? 0. : matches.front().sharedEnergy / objWeight;
    purity_->Fill(bestPurity);

    int sharedBranches = 0;
    for (auto const& m : matches)
      if (m.sharedEnergy / objWeight >= mergeThreshold_)
        ++sharedBranches;
    if (sharedBranches >= 2) {
      mergeNumEta_->Fill(obj.eta);
      mergeNumX_->Fill(obj.x);
    }

    if (bestPurity >= matchThreshold_)
      ++recoMatchCount[matches.front().rootParticleId];
    else {
      fakeNumEta_->Fill(obj.eta);
      fakeNumX_->Fill(obj.x);
    }
  }

  // Sim -> reco: efficiency / duplicate over the selected branches that carry hits.
  const uint32_t nP = graph.nParticles();
  auto considerRoot = [&](uint32_t r) {
    if (interestingPdgIds_.empty())
      return true;
    return std::find(interestingPdgIds_.begin(), interestingPdgIds_.end(), graph.particles[r].pdgId) !=
           interestingPdgIds_.end();
  };
  for (uint32_t r = 0; r < nP; ++r) {
    if (!considerRoot(r) || channelHits(hitIndex, r).empty())
      continue;
    auto const& p = graph.particles[r].momentum;
    const double eta = p.eta();
    const double x = Traits::particleX(p);
    if (!selected(eta, x))
      continue;
    denomEta_->Fill(eta);
    denomX_->Fill(x);
    auto it = recoMatchCount.find(r);
    const int n = it != recoMatchCount.end() ? it->second : 0;
    if (n >= 1) {
      effNumEta_->Fill(eta);
      effNumX_->Fill(x);
    }
    if (n >= 2) {
      dupNumEta_->Fill(eta);
      dupNumX_->Fill(x);
    }
  }
}

template <class Traits>
void BranchRecoValidatorT<Traits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<std::vector<int>>("interestingPdgIds", {})
      ->setComment("Restrict the branch side to these PDG ids (empty = all branches).");
  desc.add<std::string>("folder", "BranchValidator/Reco");
  desc.add<std::string>("xName", "pt")->setComment("Second-axis ME name suffix (e.g. pt or energy).");
  desc.add<std::string>("xTitle", "p_{T} [GeV]");
  desc.add<double>("xMax", 200.);
  desc.add<double>("minX", 0.);
  desc.add<double>("minAbsEta", 0.);
  desc.add<double>("maxAbsEta", 3.0);
  desc.add<double>("matchThreshold", 0.5)->setComment("Min best-branch purity for a reco object to count as matched.");
  desc.add<double>("mergeThreshold", 0.3)
      ->setComment("Min shared fraction for a branch to count toward a merge (>=2 -> merged reco object).");
  Traits::fillDescriptions(desc);
  descriptions.addWithDefaultLabel(desc);
}

using BranchTrackRecoValidator = BranchRecoValidatorT<TrackRecoTraits>;
using BranchTracksterRecoValidator = BranchRecoValidatorT<TracksterRecoTraits>;
DEFINE_FWK_MODULE(BranchTrackRecoValidator);
DEFINE_FWK_MODULE(BranchTracksterRecoValidator);
