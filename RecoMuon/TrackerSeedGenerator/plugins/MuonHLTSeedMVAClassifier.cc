
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
  edm::EDGetTokenT<TrajectorySeedCollection> t_Seed_;
  edm::EDGetTokenT<l1t::MuonBxCollection> t_L1Muon_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> t_L2Muon_;

  typedef std::pair<std::unique_ptr<const SeedMvaEstimator>, std::unique_ptr<const SeedMvaEstimator>> pairSeedMvaEstimator;
  pairSeedMvaEstimator mvaEstimator_;

  edm::FileInPath mvaFileB_;
  edm::FileInPath mvaFileE_;

  std::vector<double> mvaScaleMeanB_;
  std::vector<double> mvaScaleStdB_;
  std::vector<double> mvaScaleMeanE_;
  std::vector<double> mvaScaleStdE_;

  const double etaEdge_;
  const double mvaCutB_;
  const double mvaCutE_;

  const bool doSort_;
  const int nSeedsMaxB_;
  const int nSeedsMaxE_;

  const bool rejectAll_;
  const bool isFromL1_;
  const int minL1Qual_;
  const double baseScore_;

  double getSeedMva(const pairSeedMvaEstimator& pairMvaEstimator,
                    const TrajectorySeed& seed,
                    GlobalVector global_p,
                    edm::Handle<l1t::MuonBxCollection>& h_L1Muon,
                    int minL1Qual_,
                    edm::Handle<reco::RecoChargedCandidateCollection>& h_L2Muon,
                    bool isFromL1_,
                    double baseScore_);
};

MuonHLTSeedMVAClassifier::MuonHLTSeedMVAClassifier(const edm::ParameterSet& iConfig)
    : t_Seed_(consumes<TrajectorySeedCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      t_L1Muon_(consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("L1Muon"))),
      t_L2Muon_(consumes<reco::RecoChargedCandidateCollection>(iConfig.getParameter<edm::InputTag>("L2Muon"))),

      etaEdge_(iConfig.getParameter<double>("etaEdge")),
      mvaCutB_(iConfig.getParameter<double>("mvaCutB")),
      mvaCutE_(iConfig.getParameter<double>("mvaCutE")),

      doSort_(iConfig.getParameter<bool>("doSort")),
      nSeedsMaxB_(iConfig.getParameter<int>("nSeedsMaxB")),
      nSeedsMaxE_(iConfig.getParameter<int>("nSeedsMaxE")),

      rejectAll_(iConfig.getParameter<bool>("rejectAll")),
      isFromL1_(iConfig.getParameter<bool>("isFromL1")),
      minL1Qual_(iConfig.getParameter<int>("minL1Qual")),
      baseScore_(iConfig.getParameter<double>("baseScore")) {

  if (!rejectAll_) {
    mvaFileB_ = iConfig.getUntrackedParameter<edm::FileInPath>("mvaFileB");
    mvaFileE_ = iConfig.getUntrackedParameter<edm::FileInPath>("mvaFileE");

    mvaScaleMeanB_ = iConfig.getUntrackedParameter<std::vector<double>>("mvaScaleMeanB");
    mvaScaleStdB_ = iConfig.getUntrackedParameter<std::vector<double>>("mvaScaleStdB");
    mvaScaleMeanE_ = iConfig.getUntrackedParameter<std::vector<double>>("mvaScaleMeanE");
    mvaScaleStdE_ = iConfig.getUntrackedParameter<std::vector<double>>("mvaScaleStdE");

    mvaEstimator_ = std::make_pair(std::make_unique<SeedMvaEstimator>(mvaFileB_, mvaScaleMeanB_, mvaScaleStdB_),
                                   std::make_unique<SeedMvaEstimator>(mvaFileE_, mvaScaleMeanE_, mvaScaleStdE_));
  }

  produces<TrajectorySeedCollection>();
}

// -- method called on each new Event
void MuonHLTSeedMVAClassifier::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  edm::ESHandle<TrackerGeometry> trkGeom;
  iSetup.get<TrackerDigiGeometryRecord>().get(trkGeom);

  edm::Handle<l1t::MuonBxCollection> h_L1Muon;
  iEvent.getByToken(t_L1Muon_, h_L1Muon);

  edm::Handle<reco::RecoChargedCandidateCollection> h_L2Muon;
  if(!isFromL1_) {
    iEvent.getByToken(t_L2Muon_, h_L2Muon);
  }

  edm::Handle<TrajectorySeedCollection> h_Seed;
  iEvent.getByToken(t_Seed_, h_Seed);

  std::vector<std::pair<unsigned, double>> pairSeedIdxMvaScoreB = {};
  std::vector<std::pair<unsigned, double>> pairSeedIdxMvaScoreE = {};
  for (auto i = 0U; i < h_Seed->size(); ++i) {
    const auto& seed(h_Seed->at(i));

    GlobalVector global_p = trkGeom->idToDet(seed.startingState().detId())
                                   ->surface().toGlobal(seed.startingState().parameters().momentum());

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

    double mva =
        getSeedMva(mvaEstimator_, seed, global_p, h_L1Muon, minL1Qual_, h_L2Muon, isFromL1_, baseScore_);

    double score = 1. / (1. + std::exp(-1. * mva));
    bool passMva = ((isB && (score > mvaCutB_)) || (!isB && (score > mvaCutE_)));
    if (!passMva)
      continue;

    if (doSort_) {
      if (isB)
        pairSeedIdxMvaScoreB.push_back(std::make_pair(i, score));
      else
        pairSeedIdxMvaScoreE.push_back(std::make_pair(i, score));
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
      const auto& seed(h_Seed->at(pairSeedIdxMvaScoreB.at(i).first));
      result->emplace_back(seed);
    }

    for (auto i = 0U; i < pairSeedIdxMvaScoreE.size(); ++i) {
      if ((int)i == nSeedsMaxE_)
        break;
      const auto& seed(h_Seed->at(pairSeedIdxMvaScoreE.at(i).first));
      result->emplace_back(seed);
    }
  }

  iEvent.put(std::move(result));
}

double MuonHLTSeedMVAClassifier::getSeedMva(const pairSeedMvaEstimator& pairMvaEstimator,
                                            const TrajectorySeed& seed,
                                            GlobalVector global_p,
                                            edm::Handle<l1t::MuonBxCollection>& h_L1Muon,
                                            int minL1Qual_,
                                            edm::Handle<reco::RecoChargedCandidateCollection>& h_L2Muon,
                                            bool isFromL1_,
                                            double baseScore_) {
  double mva = 0.;
  if (std::abs(global_p.eta()) < etaEdge_) {
    mva = pairMvaEstimator.first->computeMva(
        seed, global_p, h_L1Muon, minL1Qual_, h_L2Muon, isFromL1_);
  } else {
    mva = pairMvaEstimator.second->computeMva(
        seed, global_p, h_L1Muon, minL1Qual_, h_L2Muon, isFromL1_);
  }

  return (mva + baseScore_);
}

// -- method fills 'descriptions' with the allowed parameters for the module  ------------
void MuonHLTSeedMVAClassifier::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltIter2IterL3MuonPixelSeeds", ""));
  desc.add<edm::InputTag>("L1Muon", edm::InputTag("hltGtStage2Digis", "Muon"));
  desc.add<edm::InputTag>("L2Muon", edm::InputTag("hltL2MuonCandidates", ""));

  desc.add<double>("etaEdge", 1.2);
  desc.add<double>("mvaCutB", -1.);
  desc.add<double>("mvaCutE", -1.);

  desc.add<bool>("doSort", false);
  desc.add<int>("nSeedsMaxB", 1e6);
  desc.add<int>("nSeedsMaxE", 1e6);

  desc.add<bool>("rejectAll", false);
  desc.add<bool>("isFromL1", false);
  desc.add<int>("minL1Qual", 7);
  desc.add<double>("baseScore", 0.5);

  desc.addOptionalUntracked<edm::FileInPath>("mvaFileB", edm::FileInPath("RecoMuon/TrackerSeedGenerator/data/Run3v6_Barrel_hltIter2.xml"));
  desc.addOptionalUntracked<edm::FileInPath>("mvaFileE", edm::FileInPath("RecoMuon/TrackerSeedGenerator/data/Run3v6_Endcap_hltIter2.xml"));
  desc.addOptionalUntracked<std::vector<double>>("mvaScaleMeanB", {0.});
  desc.addOptionalUntracked<std::vector<double>>("mvaScaleStdB", {1.});
  desc.addOptionalUntracked<std::vector<double>>("mvaScaleMeanE", {0.});
  desc.addOptionalUntracked<std::vector<double>>("mvaScaleStdE", {1.});

  descriptions.add("MuonHLTSeedMVAClassifier", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonHLTSeedMVAClassifier);
