
// Package:    RecoMuon_TrackerSeedGenerator
// Class:      MuonHLTSeedMVAClassifier

// Original Author:  Won Jun, OH Minseok
//         Created:  Fri, 28 May 2021

// system include files
#include <memory>
#include <cmath>
#include <tinyxml2.h>

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

#include "CommonTools/MVAUtils/interface/TMVAZipReader.h"

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
  bool checkMVAFileConsistency(const std::string& weightsFileFullPath, bool isFromL1) const;

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

bool MuonHLTSeedMVAClassifier::checkMVAFileConsistency(const std::string& weightsFileFullPath,
                                                       const bool isFromL1) const {
  tinyxml2::XMLDocument xmlDoc;
  if (reco::details::hasEnding(weightsFileFullPath, ".xml")) {
    xmlDoc.LoadFile(weightsFileFullPath.c_str());
  } else {
    edm::LogError("MuonHLTSeedMVAClassifier") << "unsupported file extension, it should be a .xml file!";
    return false;
  }
  tinyxml2::XMLElement* root = xmlDoc.FirstChildElement("MethodSetup");
  if (root == nullptr) {
    edm::LogError("MuonHLTSeedMVAClassifier") << "could not retrieve the MethodSetup node from XML file!";
    return false;
  }

  const auto& vars = root->FirstChildElement("Variables");
  size_t n = 0;
  if (vars != nullptr) {
    for (tinyxml2::XMLElement* e = vars->FirstChildElement("Variable"); e != nullptr;
         e = e->NextSiblingElement("Variable")) {
      ++n;
    }
  } else {
    edm::LogError("MuonHLTSeedMVAClassifier") << "could not retrieve the Variables node from XML file!";
    return false;
  }

  LogTrace("MuonHLTSeedMVAClassifier") << "MVA file:" << weightsFileFullPath.c_str() << " n Var:" << n;
  bool condition = (isFromL1 && (n == inputIndexes::kLastL1)) || (!isFromL1 && (n == inputIndexes::kLastL2));
  return condition;
}

MuonHLTSeedMVAClassifier::MuonHLTSeedMVAClassifier(const edm::ParameterSet& iConfig)
    : seedToken_(consumes<TrajectorySeedCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      l1MuonToken_(consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("L1Muon"))),
      l2MuonToken_(consumes<reco::RecoChargedCandidateCollection>(iConfig.getParameter<edm::InputTag>("L2Muon"))),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      rejectAll_(iConfig.getParameter<bool>("rejectAll")),
      isFromL1_(iConfig.getParameter<bool>("isFromL1")),
      mvaFileB_(iConfig.getParameter<edm::FileInPath>("mvaFileB")),
      mvaFileE_(iConfig.getParameter<edm::FileInPath>("mvaFileE")),
      mvaScaleMeanB_(iConfig.getParameter<std::vector<double>>("mvaScaleMeanB")),
      mvaScaleStdB_(iConfig.getParameter<std::vector<double>>("mvaScaleStdB")),
      mvaScaleMeanE_(iConfig.getParameter<std::vector<double>>("mvaScaleMeanE")),
      mvaScaleStdE_(iConfig.getParameter<std::vector<double>>("mvaScaleStdE")),
      doSort_(iConfig.getParameter<bool>("doSort")),
      nSeedsMaxB_(iConfig.getParameter<int>("nSeedsMaxB")),
      nSeedsMaxE_(iConfig.getParameter<int>("nSeedsMaxE")),
      etaEdge_(iConfig.getParameter<double>("etaEdge")),
      mvaCutB_(iConfig.getParameter<double>("mvaCutB")),
      mvaCutE_(iConfig.getParameter<double>("mvaCutE")),
      minL1Qual_(iConfig.getParameter<int>("minL1Qual")),
      baseScore_(iConfig.getParameter<double>("baseScore")) {
  const auto& mvaFileBPath = mvaFileB_.fullPath();
  const auto& mvaFileEPath = mvaFileE_.fullPath();

  if (!checkMVAFileConsistency(mvaFileBPath, isFromL1_) || !checkMVAFileConsistency(mvaFileEPath, isFromL1_)) {
    throw cms::Exception("ConfigurationError") << " MVA files appear to be not consistent with the value of isFromL1 "
                                                  "parameter.\n Please check your configuration.";
  }

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
