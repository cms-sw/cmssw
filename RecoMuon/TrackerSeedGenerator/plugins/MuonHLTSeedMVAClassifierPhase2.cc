// Package:    RecoMuon_TrackerSeedGenerator
// Class:      MuonHLTSeedMVAClassifierPhase2

// Original Author:  OH Minseok, Sungwon Kim, Won Jun
//         Created:  Mon, 08 Jun 2020 06:20:44 GMT

// system include files
#include <memory>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

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

// -- for L1TkMu propagation
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTrackerBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoMuon/TrackerSeedGenerator/interface/SeedMvaEstimatorPhase2.h"

// class declaration
bool sortByMvaScorePhase2(const std::pair<unsigned, double>& A, const std::pair<unsigned, double>& B) {
  return (A.second > B.second);
};

class MuonHLTSeedMVAClassifierPhase2 : public edm::stream::EDProducer<> {
public:
  explicit MuonHLTSeedMVAClassifierPhase2(const edm::ParameterSet&);
  ~MuonHLTSeedMVAClassifierPhase2() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<TrajectorySeedCollection> t_Seed_;
  const edm::EDGetTokenT<l1t::TrackerMuonCollection> t_L1TkMu_;

  typedef std::pair<std::unique_ptr<const SeedMvaEstimatorPhase2>, std::unique_ptr<const SeedMvaEstimatorPhase2>>
      pairSeedMvaEstimator;
  pairSeedMvaEstimator mvaEstimator_;

  const edm::FileInPath mvaFile_B_0_;
  const edm::FileInPath mvaFile_E_0_;

  const std::vector<double> mvaScaleMean_B_;
  const std::vector<double> mvaScaleStd_B_;
  const std::vector<double> mvaScaleMean_E_;
  const std::vector<double> mvaScaleStd_E_;

  const double etaEdge_;
  const double mvaCut_B_;
  const double mvaCut_E_;

  const bool doSort_;
  const int nSeedsMax_B_;
  const int nSeedsMax_E_;

  const double baseScore_;

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyESToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryESToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldESToken_;
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetESToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorESToken_;

  double getSeedMva(const pairSeedMvaEstimator& pairMvaEstimator,
                    const TrajectorySeed& seed,
                    const GlobalVector& global_p,
                    const GlobalPoint& global_x,
                    const edm::Handle<l1t::TrackerMuonCollection>& h_L1TkMu,
                    const edm::ESHandle<MagneticField>& magfieldH,
                    const Propagator& propagatorAlong,
                    const GeometricSearchTracker& geomTracker);
};

// constructors and destructor
MuonHLTSeedMVAClassifierPhase2::MuonHLTSeedMVAClassifierPhase2(const edm::ParameterSet& iConfig)
    : t_Seed_(consumes<TrajectorySeedCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      t_L1TkMu_(consumes<l1t::TrackerMuonCollection>(iConfig.getParameter<edm::InputTag>("L1TkMu"))),

      mvaFile_B_0_(iConfig.getParameter<edm::FileInPath>("mvaFile_B_0")),
      mvaFile_E_0_(iConfig.getParameter<edm::FileInPath>("mvaFile_E_0")),

      mvaScaleMean_B_(iConfig.getParameter<std::vector<double>>("mvaScaleMean_B")),
      mvaScaleStd_B_(iConfig.getParameter<std::vector<double>>("mvaScaleStd_B")),
      mvaScaleMean_E_(iConfig.getParameter<std::vector<double>>("mvaScaleMean_E")),
      mvaScaleStd_E_(iConfig.getParameter<std::vector<double>>("mvaScaleStd_E")),

      etaEdge_(iConfig.getParameter<double>("etaEdge")),
      mvaCut_B_(iConfig.getParameter<double>("mvaCut_B")),
      mvaCut_E_(iConfig.getParameter<double>("mvaCut_E")),

      doSort_(iConfig.getParameter<bool>("doSort")),
      nSeedsMax_B_(iConfig.getParameter<int>("nSeedsMax_B")),
      nSeedsMax_E_(iConfig.getParameter<int>("nSeedsMax_E")),

      baseScore_(iConfig.getParameter<double>("baseScore")),

      trackerTopologyESToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      trackerGeometryESToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      magFieldESToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      geomDetESToken_(esConsumes<GeometricDet, IdealGeometryRecord>()),
      propagatorESToken_(
          esConsumes<Propagator, TrackingComponentsRecord>(edm::ESInputTag("", "PropagatorWithMaterialParabolicMf"))) {
  produces<TrajectorySeedCollection>();

  mvaEstimator_ =
      std::make_pair(std::make_unique<SeedMvaEstimatorPhase2>(mvaFile_B_0_, mvaScaleMean_B_, mvaScaleStd_B_),
                     std::make_unique<SeedMvaEstimatorPhase2>(mvaFile_E_0_, mvaScaleMean_E_, mvaScaleStd_E_));
}

// member functions
// ------------ method called on each new Event  ------------
void MuonHLTSeedMVAClassifierPhase2::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto result = std::make_unique<TrajectorySeedCollection>();

  edm::ESHandle<TrackerGeometry> trkGeom = iSetup.getHandle(trackerGeometryESToken_);
  edm::ESHandle<GeometricDet> geomDet = iSetup.getHandle(geomDetESToken_);
  edm::ESHandle<TrackerTopology> trkTopo = iSetup.getHandle(trackerTopologyESToken_);

  GeometricSearchTrackerBuilder builder;
  GeometricSearchTracker geomTracker = *(builder.build(&(*geomDet), &(*trkGeom), &(*trkTopo)));

  edm::Handle<l1t::TrackerMuonCollection> h_L1TkMu;
  bool hasL1TkMu = iEvent.getByToken(t_L1TkMu_, h_L1TkMu);

  edm::Handle<TrajectorySeedCollection> h_Seed;
  bool hasSeed = iEvent.getByToken(t_Seed_, h_Seed);

  edm::ESHandle<MagneticField> magfieldH = iSetup.getHandle(magFieldESToken_);
  edm::ESHandle<Propagator> propagatorAlongH = iSetup.getHandle(propagatorESToken_);
  std::unique_ptr<Propagator> propagatorAlong = SetPropagationDirection(*propagatorAlongH, alongMomentum);

  if (!(hasL1TkMu && hasSeed)) {
    edm::LogError("SeedClassifierError") << "Error! Cannot find L1TkMuon or TrajectorySeed\n"
                                         << "hasL1TkMu : " << hasL1TkMu << "\n"
                                         << "hasSeed : " << hasSeed << "\n";
    return;
  }

  // -- sort seeds by MVA score and chooes top nSeedsMax_B_ / nSeedsMax_E_
  if (doSort_) {
    std::vector<std::pair<unsigned, double>> pairSeedIdxMvaScore_B = {};
    std::vector<std::pair<unsigned, double>> pairSeedIdxMvaScore_E = {};

    for (auto i = 0U; i < h_Seed->size(); ++i) {
      const auto& seed(h_Seed->at(i));

      GlobalVector global_p = trkGeom->idToDet(seed.startingState().detId())
                                  ->surface()
                                  .toGlobal(seed.startingState().parameters().momentum());
      GlobalPoint global_x = trkGeom->idToDet(seed.startingState().detId())
                                 ->surface()
                                 .toGlobal(seed.startingState().parameters().position());

      bool isB = (std::abs(global_p.eta()) < etaEdge_);

      if (isB && nSeedsMax_B_ < 0) {
        result->emplace_back(seed);
        continue;
      }

      if (!isB && nSeedsMax_E_ < 0) {
        result->emplace_back(seed);
        continue;
      }

      double mva = getSeedMva(
          mvaEstimator_, seed, global_p, global_x, h_L1TkMu, magfieldH, *(propagatorAlong.get()), geomTracker);

      double logistic = 1 / (1 + std::exp(-mva));

      if (isB)
        pairSeedIdxMvaScore_B.push_back(make_pair(i, logistic));
      else
        pairSeedIdxMvaScore_E.push_back(make_pair(i, logistic));
    }

    std::sort(pairSeedIdxMvaScore_B.begin(), pairSeedIdxMvaScore_B.end(), sortByMvaScorePhase2);
    std::sort(pairSeedIdxMvaScore_E.begin(), pairSeedIdxMvaScore_E.end(), sortByMvaScorePhase2);

    for (auto i = 0U; i < pairSeedIdxMvaScore_B.size(); ++i) {
      if ((int)i == nSeedsMax_B_)
        break;
      const auto& seed(h_Seed->at(pairSeedIdxMvaScore_B.at(i).first));
      result->emplace_back(seed);
    }

    for (auto i = 0U; i < pairSeedIdxMvaScore_E.size(); ++i) {
      if ((int)i == nSeedsMax_E_)
        break;
      const auto& seed(h_Seed->at(pairSeedIdxMvaScore_E.at(i).first));
      result->emplace_back(seed);
    }
  }

  // -- simple fitering based on Mva threshold
  else {
    for (auto i = 0U; i < h_Seed->size(); ++i) {
      const auto& seed(h_Seed->at(i));

      GlobalVector global_p = trkGeom->idToDet(seed.startingState().detId())
                                  ->surface()
                                  .toGlobal(seed.startingState().parameters().momentum());
      GlobalPoint global_x = trkGeom->idToDet(seed.startingState().detId())
                                 ->surface()
                                 .toGlobal(seed.startingState().parameters().position());

      bool isB = (std::abs(global_p.eta()) < etaEdge_);

      if (isB && mvaCut_B_ <= 0.) {
        result->emplace_back(seed);
        continue;
      }

      if (!isB && mvaCut_E_ <= 0.) {
        result->emplace_back(seed);
        continue;
      }

      double mva = getSeedMva(
          mvaEstimator_, seed, global_p, global_x, h_L1TkMu, magfieldH, *(propagatorAlong.get()), geomTracker);

      double logistic = 1 / (1 + std::exp(-mva));

      bool passMva = ((isB && (logistic > mvaCut_B_)) || (!isB && (logistic > mvaCut_E_)));

      if (passMva)
        result->emplace_back(seed);
    }
  }

  iEvent.put(std::move(result));
}

double MuonHLTSeedMVAClassifierPhase2::getSeedMva(const pairSeedMvaEstimator& pairMvaEstimator,
                                                  const TrajectorySeed& seed,
                                                  const GlobalVector& global_p,
                                                  const GlobalPoint& global_x,
                                                  const edm::Handle<l1t::TrackerMuonCollection>& h_L1TkMu,
                                                  const edm::ESHandle<MagneticField>& magfieldH,
                                                  const Propagator& propagatorAlong,
                                                  const GeometricSearchTracker& geomTracker) {
  double mva = 0.;

  if (std::abs(global_p.eta()) < etaEdge_) {
    mva =
        pairMvaEstimator.first->computeMva(seed, global_p, global_x, h_L1TkMu, magfieldH, propagatorAlong, geomTracker);
  } else {
    mva = pairMvaEstimator.second->computeMva(
        seed, global_p, global_x, h_L1TkMu, magfieldH, propagatorAlong, geomTracker);
  }

  return (baseScore_ + mva);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MuonHLTSeedMVAClassifierPhase2::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltIter2Phase2L3FromL1TkMuonPixelSeeds", ""));
  desc.add<edm::InputTag>("L1TkMu", edm::InputTag("L1TkMuons", "Muon"));
  desc.add<edm::FileInPath>("mvaFile_B_0",
                            edm::FileInPath("RecoMuon/TrackerSeedGenerator/data/xgb_Phase2_Iter2FromL1_barrel_v0.xml"));
  desc.add<edm::FileInPath>("mvaFile_E_0",
                            edm::FileInPath("RecoMuon/TrackerSeedGenerator/data/xgb_Phase2_Iter2FromL1_endcap_v0.xml"));
  desc.add<std::vector<double>>("mvaScaleMean_B",
                                {0.00033113700731766336,
                                 1.6825601468762878e-06,
                                 1.790932122524803e-06,
                                 0.010534608406382916,
                                 0.005969459957330139,
                                 0.0009605022254971113,
                                 0.04384189672781466,
                                 7.846741237608237e-05,
                                 0.40725050850004824,
                                 0.41125151617410227,
                                 0.39815551065544846});
  desc.add<std::vector<double>>("mvaScaleStd_B",
                                {0.0006042948363798624,
                                 2.445644111872427e-06,
                                 3.454992543447134e-06,
                                 0.09401581628887255,
                                 0.7978806947573766,
                                 0.4932933044535928,
                                 0.04180518265631776,
                                 0.058296511682094855,
                                 0.4071857009373577,
                                 0.41337782307392973,
                                 0.4101160349549534});
  desc.add<std::vector<double>>("mvaScaleMean_E",
                                {0.00022658482374555603,
                                 5.358921973784045e-07,
                                 1.010003713549798e-06,
                                 0.0007886873612224615,
                                 0.001197730548842408,
                                 -0.0030252353426003594,
                                 0.07151944804171254,
                                 -0.0006940626775109026,
                                 0.20535152195939896,
                                 0.2966816533783824,
                                 0.28798220230180455});
  desc.add<std::vector<double>>("mvaScaleStd_E",
                                {0.0003857726789049956,
                                 1.4853721474087994e-06,
                                 6.982997036736564e-06,
                                 0.04071340757666084,
                                 0.5897606560095399,
                                 0.33052121398064654,
                                 0.05589386786541949,
                                 0.08806273533388546,
                                 0.3254586902665612,
                                 0.3293354496231377,
                                 0.3179899794578072});
  desc.add<bool>("doSort", true);
  desc.add<int>("nSeedsMax_B", 20);
  desc.add<int>("nSeedsMax_E", 20);
  desc.add<double>("mvaCut_B", 0.);
  desc.add<double>("mvaCut_E", 0.);
  desc.add<double>("etaEdge", 1.2);
  desc.add<double>("baseScore", 0.5);
  descriptions.add("MuonHLTSeedMVAClassifierPhase2", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(MuonHLTSeedMVAClassifierPhase2);