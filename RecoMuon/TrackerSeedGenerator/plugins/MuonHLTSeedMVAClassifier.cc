
// Package:    RecoMuon_TrackerSeedGenerator
// Class:      MuonHLTSeedMVAClassifier

// Original Author:  Won Jun, OH Minseok
//         Created:  Fri, 28 May 2021

// system include files
#include <memory>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

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
class MuonHLTSeedMVAClassifier : public edm::global::EDProducer<> {
public:
  explicit MuonHLTSeedMVAClassifier(const edm::ParameterSet&);
  ~MuonHLTSeedMVAClassifier() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  enum Region { kBarrel, kEndcap };

  // read the barrel or endcap estimator from its XML file, and check that its number of input variables is
  // consistent with isFromL1; empty if rejectAll is set
  static std::unique_ptr<const SeedMvaEstimator> makeMvaEstimator(const edm::ParameterSet&, Region region);

  // member data
  const edm::EDGetTokenT<TrajectorySeedCollection> seedToken_;
  const edm::EDGetTokenT<l1t::MuonBxCollection> l1MuonToken_;
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> l2MuonToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;

  // read once per job and shared by all streams
  const std::unique_ptr<const SeedMvaEstimator> mvaEstimatorB_;
  const std::unique_ptr<const SeedMvaEstimator> mvaEstimatorE_;

  const bool rejectAll_;

  const bool doSort_;
  const int nSeedsMaxB_;
  const int nSeedsMaxE_;

  const double etaEdge_;
  const double mvaCutB_;
  const double mvaCutE_;

  const double baseScore_;

  double getSeedMva(const TrajectorySeed& seed,
                    const GlobalVector& global_p,
                    const l1t::MuonBxCollection& l1Muons,
                    const reco::RecoChargedCandidateCollection& l2Muons) const;
};

std::unique_ptr<const SeedMvaEstimator> MuonHLTSeedMVAClassifier::makeMvaEstimator(const edm::ParameterSet& iConfig,
                                                                                   Region region) {
  // suffix of the parameter names of this region
  const std::string suffix = (region == kBarrel) ? "B" : "E";

  if (iConfig.getParameter<bool>("rejectAll"))
    return nullptr;

  const bool isFromL1 = iConfig.getParameter<bool>("isFromL1");
  auto estimator =
      std::make_unique<SeedMvaEstimator>(iConfig.getParameter<edm::FileInPath>("mvaFile" + suffix),
                                         iConfig.getParameter<std::vector<double>>("mvaScaleMean" + suffix),
                                         iConfig.getParameter<std::vector<double>>("mvaScaleStd" + suffix),
                                         isFromL1,
                                         iConfig.getParameter<int>("minL1Qual"));

  const unsigned int expected = isFromL1 ? inputIndexes::kLastL1 : inputIndexes::kLastL2;
  if (estimator->nVariables() != expected) {
    throw cms::Exception("ConfigurationError") << " MVA files appear to be not consistent with the value of isFromL1 "
                                                  "parameter.\n Please check your configuration.";
  }
  return estimator;
}

MuonHLTSeedMVAClassifier::MuonHLTSeedMVAClassifier(const edm::ParameterSet& iConfig)
    : seedToken_(consumes<TrajectorySeedCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      l1MuonToken_(consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("L1Muon"))),
      l2MuonToken_(consumes<reco::RecoChargedCandidateCollection>(iConfig.getParameter<edm::InputTag>("L2Muon"))),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      mvaEstimatorB_(makeMvaEstimator(iConfig, kBarrel)),
      mvaEstimatorE_(makeMvaEstimator(iConfig, kEndcap)),
      rejectAll_(iConfig.getParameter<bool>("rejectAll")),
      doSort_(iConfig.getParameter<bool>("doSort")),
      nSeedsMaxB_(iConfig.getParameter<int>("nSeedsMaxB")),
      nSeedsMaxE_(iConfig.getParameter<int>("nSeedsMaxE")),
      etaEdge_(iConfig.getParameter<double>("etaEdge")),
      mvaCutB_(iConfig.getParameter<double>("mvaCutB")),
      mvaCutE_(iConfig.getParameter<double>("mvaCutE")),
      baseScore_(iConfig.getParameter<double>("baseScore")) {
  produces<TrajectorySeedCollection>();
}

// -- method called on each new Event
void MuonHLTSeedMVAClassifier::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const& iEventSetup) const {
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

    double mva = getSeedMva(seed, global_p, l1Muons, l2Muons);

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
    // sort the seeds by decreasing MVA score
    auto sortByMvaScore = [](const std::pair<unsigned, double>& a, const std::pair<unsigned, double>& b) {
      return a.second > b.second;
    };
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

double MuonHLTSeedMVAClassifier::getSeedMva(const TrajectorySeed& seed,
                                            const GlobalVector& global_p,
                                            const l1t::MuonBxCollection& l1Muons,
                                            const reco::RecoChargedCandidateCollection& l2Muons) const {
  double mva = 0.;
  if (std::abs(global_p.eta()) < etaEdge_) {
    mva = mvaEstimatorB_->computeMva(seed, global_p, l1Muons, l2Muons);
  } else {
    mva = mvaEstimatorE_->computeMva(seed, global_p, l1Muons, l2Muons);
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

  desc.add<edm::FileInPath>("mvaFileB",
                            edm::FileInPath("RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter2FromL1Seeds_barrel.xml"));
  desc.add<edm::FileInPath>("mvaFileE",
                            edm::FileInPath("RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter2FromL1Seeds_endcap.xml"));
  desc.add<std::vector<double>>("mvaScaleMeanB", {0., 0., 0., 0., 0., 0., 0., 0.});
  desc.add<std::vector<double>>("mvaScaleStdB", {1., 1., 1., 1., 1., 1., 1., 1.});
  desc.add<std::vector<double>>("mvaScaleMeanE", {0., 0., 0., 0., 0., 0., 0., 0.});
  desc.add<std::vector<double>>("mvaScaleStdE", {1., 1., 1., 1., 1., 1., 1., 1.});

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
