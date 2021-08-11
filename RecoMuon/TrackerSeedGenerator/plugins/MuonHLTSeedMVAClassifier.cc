
// Package:    RecoMuon_TrackerSeedGenerator
// Class:      MuonHLTSeedMVAClassifier

// Original Author:  Won Jun, OH Minseok
//         Created:  Fri, 28 May 2021

// system include files
#include <memory>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// TrajectorySeed
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "RecoMuon/TrackerSeedGenerator/interface/SeedMvaEstimator.h"

// class declaration
bool sortByMvaScore(const std::pair<unsigned, double>& A, const std::pair<unsigned, double>& B) {
  return (A.second > B.second);
};

class MuonHLTSeedMVAClassifier : public edm::stream::EDProducer<> {
public:
  explicit MuonHLTSeedMVAClassifier(const edm::ParameterSet&);
  ~MuonHLTSeedMVAClassifier() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // member data
  const edm::EDGetTokenT<TrajectorySeedCollection> seedToken_;
  const edm::EDGetTokenT<l1t::MuonBxCollection> l1MuonToken_;
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> l2MuonToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;

  typedef std::pair<std::unique_ptr<const SeedMvaEstimator>, std::unique_ptr<const SeedMvaEstimator>>
      PairSeedMvaEstimator;
  PairSeedMvaEstimator mvaEstimator_;

  const bool rejectAll_;
  const bool isFromL1_;

  const edm::FileInPath mvaFileB_;
  const edm::FileInPath mvaFileE_;

  const std::vector<double> mvaScaleMeanB_;
  const std::vector<double> mvaScaleStdB_;
  const std::vector<double> mvaScaleMeanE_;
  const std::vector<double> mvaScaleStdE_;

  const bool doSort_;
  const int nSeedsMaxB_;
  const int nSeedsMaxE_;

  const double etaEdge_;
  const double mvaCutB_;
  const double mvaCutE_;

  const int minL1Qual_;
  const double baseScore_;

  double getSeedMva(const PairSeedMvaEstimator& pairMvaEstimator,
                    const TrajectorySeed& seed,
                    const GlobalVector& global_p,
                    const l1t::MuonBxCollection& l1Muons,
                    const reco::RecoChargedCandidateCollection& l2Muons);
};

MuonHLTSeedMVAClassifier::MuonHLTSeedMVAClassifier(const edm::ParameterSet& iConfig)
    : seedToken_(consumes<TrajectorySeedCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      l1MuonToken_(consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("L1Muon"))),
      l2MuonToken_(consumes<reco::RecoChargedCandidateCollection>(iConfig.getParameter<edm::InputTag>("L2Muon"))),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),

      rejectAll_(iConfig.getParameter<bool>("rejectAll")),
      isFromL1_(iConfig.getParameter<bool>("isFromL1")),

      mvaFileB_(iConfig.getParameter<edm::FileInPath>(isFromL1_ ? "mvaFileBL1" : "mvaFileBL2")),
      mvaFileE_(iConfig.getParameter<edm::FileInPath>(isFromL1_ ? "mvaFileEL1" : "mvaFileEL2")),

      mvaScaleMeanB_(iConfig.getParameter<std::vector<double>>(isFromL1_ ? "mvaScaleMeanBL1" : "mvaScaleMeanBL2")),
      mvaScaleStdB_(iConfig.getParameter<std::vector<double>>(isFromL1_ ? "mvaScaleStdBL1" : "mvaScaleStdBL2")),
      mvaScaleMeanE_(iConfig.getParameter<std::vector<double>>(isFromL1_ ? "mvaScaleMeanEL1" : "mvaScaleMeanEL2")),
      mvaScaleStdE_(iConfig.getParameter<std::vector<double>>(isFromL1_ ? "mvaScaleStdEL1" : "mvaScaleStdEL2")),

      doSort_(iConfig.getParameter<bool>("doSort")),
      nSeedsMaxB_(iConfig.getParameter<int>("nSeedsMaxB")),
      nSeedsMaxE_(iConfig.getParameter<int>("nSeedsMaxE")),

      etaEdge_(iConfig.getParameter<double>("etaEdge")),
      mvaCutB_(iConfig.getParameter<double>("mvaCutB")),
      mvaCutE_(iConfig.getParameter<double>("mvaCutE")),

      minL1Qual_(iConfig.getParameter<int>("minL1Qual")),
      baseScore_(iConfig.getParameter<double>("baseScore")) {
  if (!rejectAll_) {
    mvaEstimator_ = std::make_pair(
        std::make_unique<SeedMvaEstimator>(mvaFileB_, mvaScaleMeanB_, mvaScaleStdB_, isFromL1_, minL1Qual_),
        std::make_unique<SeedMvaEstimator>(mvaFileE_, mvaScaleMeanE_, mvaScaleStdE_, isFromL1_, minL1Qual_));
  }

  produces<TrajectorySeedCollection>();
}

// -- method called on each new Event
void MuonHLTSeedMVAClassifier::produce(edm::Event& iEvent, edm::EventSetup const& iEventSetup) {
  auto result = std::make_unique<TrajectorySeedCollection>();

  if (rejectAll_) {
    iEvent.put(std::move(result));
    return;
  }

  if (doSort_ && nSeedsMaxB_ <= 0 && nSeedsMaxE_ <= 0) {
    iEvent.put(std::move(result));
    return;
  }

  if (!doSort_ && mvaCutB_ > 1. && mvaCutE_ > 1.) {
    iEvent.put(std::move(result));
    return;
  }

  const TrajectorySeedCollection& seeds = iEvent.get(seedToken_);
  const l1t::MuonBxCollection& l1Muons = iEvent.get(l1MuonToken_);
  const reco::RecoChargedCandidateCollection& l2Muons = iEvent.get(l2MuonToken_);
  const TrackerGeometry& trkGeom = iEventSetup.getData(trackerGeometryToken_);

  std::vector<std::pair<unsigned, double>> pairSeedIdxMvaScoreB = {};
  std::vector<std::pair<unsigned, double>> pairSeedIdxMvaScoreE = {};
  for (auto& seed : seeds) {
    const GlobalVector global_p =
        trkGeom.idToDet(seed.startingState().detId())->surface().toGlobal(seed.startingState().parameters().momentum());

    bool isB = (std::abs(global_p.eta()) < etaEdge_);

    if (doSort_) {
      if (isB) {
        if (nSeedsMaxB_ <= 0) {
          continue;
        }
      } else {
        if (nSeedsMaxE_ <= 0) {
          continue;
        }
      }
    } else {
      if (isB) {
        if (mvaCutB_ > 1.0) {
          continue;
        } else if (mvaCutB_ <= 0.) {
          result->emplace_back(seed);
          continue;
        }
      } else {
        if (mvaCutE_ > 1.0) {
          continue;
        } else if (mvaCutE_ <= 0.) {
          result->emplace_back(seed);
          continue;
        }
      }
    }

    double mva = getSeedMva(mvaEstimator_, seed, global_p, l1Muons, l2Muons);

    double score = 1. / (1. + std::exp(-1. * mva));
    bool passMva = isB ? score > mvaCutB_ : score > mvaCutE_;
    if (!passMva)
      continue;

    if (doSort_) {
      if (isB)
        pairSeedIdxMvaScoreB.push_back(std::make_pair(&seed - &seeds.at(0), score));
      else
        pairSeedIdxMvaScoreE.push_back(std::make_pair(&seed - &seeds.at(0), score));
    } else {
      result->emplace_back(seed);
    }
  }

  if (doSort_) {
    std::sort(pairSeedIdxMvaScoreB.begin(), pairSeedIdxMvaScoreB.end(), sortByMvaScore);
    std::sort(pairSeedIdxMvaScoreE.begin(), pairSeedIdxMvaScoreE.end(), sortByMvaScore);

    for (auto i = 0U; i < pairSeedIdxMvaScoreB.size(); ++i) {
      if ((int)i == nSeedsMaxB_)
        break;
      const auto& seed(seeds.at(pairSeedIdxMvaScoreB.at(i).first));
      result->emplace_back(seed);
    }

    for (auto i = 0U; i < pairSeedIdxMvaScoreE.size(); ++i) {
      if ((int)i == nSeedsMaxE_)
        break;
      const auto& seed(seeds.at(pairSeedIdxMvaScoreE.at(i).first));
      result->emplace_back(seed);
    }
  }

  iEvent.put(std::move(result));
}

double MuonHLTSeedMVAClassifier::getSeedMva(const PairSeedMvaEstimator& pairMvaEstimator,
                                            const TrajectorySeed& seed,
                                            const GlobalVector& global_p,
                                            const l1t::MuonBxCollection& l1Muons,
                                            const reco::RecoChargedCandidateCollection& l2Muons) {
  double mva = 0.;
  if (std::abs(global_p.eta()) < etaEdge_) {
    mva = pairMvaEstimator.first->computeMva(seed, global_p, l1Muons, l2Muons);
  } else {
    mva = pairMvaEstimator.second->computeMva(seed, global_p, l1Muons, l2Muons);
  }

  return (mva + baseScore_);
}

// -- method fills 'descriptions' with the allowed parameters for the module  ------------
void MuonHLTSeedMVAClassifier::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltIter2IterL3MuonPixelSeeds", ""));
  desc.add<edm::InputTag>("L1Muon", edm::InputTag("hltGtStage2Digis", "Muon"));
  desc.add<edm::InputTag>("L2Muon", edm::InputTag("hltL2MuonCandidates", ""));

  desc.add<bool>("rejectAll", false);
  desc.add<bool>("isFromL1", false);

  desc.add<edm::FileInPath>("mvaFileBL1",
                            edm::FileInPath("RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter2FromL1Seeds_barrel.xml"));
  desc.add<edm::FileInPath>("mvaFileEL1",
                            edm::FileInPath("RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter2FromL1Seeds_endcap.xml"));
  desc.add<edm::FileInPath>("mvaFileBL2",
                            edm::FileInPath("RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter2Seeds_barrel.xml"));
  desc.add<edm::FileInPath>("mvaFileEL2",
                            edm::FileInPath("RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter2Seeds_endcap.xml"));
  desc.add<std::vector<double>>("mvaScaleMeanBL1", {0., 0., 0., 0., 0., 0., 0., 0.});
  desc.add<std::vector<double>>("mvaScaleStdBL1", {1., 1., 1., 1., 1., 1., 1., 1.});
  desc.add<std::vector<double>>("mvaScaleMeanEL1", {0., 0., 0., 0., 0., 0., 0., 0.});
  desc.add<std::vector<double>>("mvaScaleStdEL1", {1., 1., 1., 1., 1., 1., 1., 1.});
  desc.add<std::vector<double>>("mvaScaleMeanBL2", {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
  desc.add<std::vector<double>>("mvaScaleStdBL2", {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});
  desc.add<std::vector<double>>("mvaScaleMeanEL2", {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
  desc.add<std::vector<double>>("mvaScaleStdEL2", {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});

  desc.add<bool>("doSort", false);
  desc.add<int>("nSeedsMaxB", 1e6);
  desc.add<int>("nSeedsMaxE", 1e6);

  desc.add<double>("etaEdge", 1.2);
  desc.add<double>("mvaCutB", -1.);
  desc.add<double>("mvaCutE", -1.);

  desc.add<int>("minL1Qual", 7);
  desc.add<double>("baseScore", 0.5);

  descriptions.add("MuonHLTSeedMVAClassifier", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonHLTSeedMVAClassifier);
