/*
 * \class L2TauTagProducer
 *
 * L2Tau identification using Convolutional NN.
 *
 * \author Valeria D'Amante, Università di Siena and INFN Pisa
 *         Konstantin Androsov, EPFL and ETHZ
*/
#include <memory>
#include <boost/property_tree/json_parser.hpp>
#include <cmath>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitDefs.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "RecoTracker/PixelTrackFitting/interface/FitUtils.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"

#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexUtilities.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"

namespace L2TauTagNNv1 {
  constexpr int nCellEta = 5;
  constexpr int nCellPhi = 5;
  constexpr int nVars = 31;
  constexpr float dR_max = 0.5;
  enum class NNInputs {
    nVertices = 0,
    l1Tau_pt,
    l1Tau_eta,
    l1Tau_hwIso,
    EcalEnergySum,
    EcalSize,
    EcalEnergyStdDev,
    EcalDeltaEta,
    EcalDeltaPhi,
    EcalChi2,
    EcalEnergySumForPositiveChi2,
    EcalSizeForPositiveChi2,
    HcalEnergySum,
    HcalSize,
    HcalEnergyStdDev,
    HcalDeltaEta,
    HcalDeltaPhi,
    HcalChi2,
    HcalEnergySumForPositiveChi2,
    HcalSizeForPositiveChi2,
    PatatrackPtSum,
    PatatrackSize,
    PatatrackSizeWithVertex,
    PatatrackPtSumWithVertex,
    PatatrackChargeSum,
    PatatrackDeltaEta,
    PatatrackDeltaPhi,
    PatatrackChi2OverNdof,
    PatatrackNdof,
    PatatrackDxy,
    PatatrackDz
  };

  const std::map<NNInputs, std::string> varNameMap = {
      {NNInputs::nVertices, "nVertices"},
      {NNInputs::l1Tau_pt, "l1Tau_pt"},
      {NNInputs::l1Tau_eta, "l1Tau_eta"},
      {NNInputs::l1Tau_hwIso, "l1Tau_hwIso"},
      {NNInputs::EcalEnergySum, "EcalEnergySum"},
      {NNInputs::EcalSize, "EcalSize"},
      {NNInputs::EcalEnergyStdDev, "EcalEnergyStdDev"},
      {NNInputs::EcalDeltaEta, "EcalDeltaEta"},
      {NNInputs::EcalDeltaPhi, "EcalDeltaPhi"},
      {NNInputs::EcalChi2, "EcalChi2"},
      {NNInputs::EcalEnergySumForPositiveChi2, "EcalEnergySumForPositiveChi2"},
      {NNInputs::EcalSizeForPositiveChi2, "EcalSizeForPositiveChi2"},
      {NNInputs::HcalEnergySum, "HcalEnergySum"},
      {NNInputs::HcalSize, "HcalSize"},
      {NNInputs::HcalEnergyStdDev, "HcalEnergyStdDev"},
      {NNInputs::HcalDeltaEta, "HcalDeltaEta"},
      {NNInputs::HcalDeltaPhi, "HcalDeltaPhi"},
      {NNInputs::HcalChi2, "HcalChi2"},
      {NNInputs::HcalEnergySumForPositiveChi2, "HcalEnergySumForPositiveChi2"},
      {NNInputs::HcalSizeForPositiveChi2, "HcalSizeForPositiveChi2"},
      {NNInputs::PatatrackPtSum, "PatatrackPtSum"},
      {NNInputs::PatatrackSize, "PatatrackSize"},
      {NNInputs::PatatrackSizeWithVertex, "PatatrackSizeWithVertex"},
      {NNInputs::PatatrackPtSumWithVertex, "PatatrackPtSumWithVertex"},
      {NNInputs::PatatrackChargeSum, "PatatrackChargeSum"},
      {NNInputs::PatatrackDeltaEta, "PatatrackDeltaEta"},
      {NNInputs::PatatrackDeltaPhi, "PatatrackDeltaPhi"},
      {NNInputs::PatatrackChi2OverNdof, "PatatrackChi2OverNdof"},
      {NNInputs::PatatrackNdof, "PatatrackNdof"},
      {NNInputs::PatatrackDxy, "PatatrackDxy"},
      {NNInputs::PatatrackDz, "PatatrackDz"}};
}  // namespace L2TauTagNNv1
namespace {
  inline float& getCellImpl(
      tensorflow::Tensor& cellGridMatrix, int tau_idx, int phi_idx, int eta_idx, L2TauTagNNv1::NNInputs NNInput_idx) {
    return cellGridMatrix.tensor<float, 4>()(tau_idx, phi_idx, eta_idx, static_cast<int>(NNInput_idx));
  }
}  // namespace
struct normDictElement {
  float mean;
  float std;
  float min;
  float max;
};

struct L2TauNNProducerCacheData {
  L2TauNNProducerCacheData() : graphDef(nullptr), session(nullptr) {}
  tensorflow::GraphDef* graphDef;
  tensorflow::Session* session;
  std::vector<normDictElement> normVec;
};

class L2TauNNProducer : public edm::stream::EDProducer<edm::GlobalCache<L2TauNNProducerCacheData>> {
public:
  using TrackSoAHost = pixelTrack::TrackSoAHostPhase1;

  struct caloRecHitCollections {
    const HBHERecHitCollection* hbhe;
    const HORecHitCollection* ho;
    const EcalRecHitCollection* eb;
    const EcalRecHitCollection* ee;
    const CaloGeometry* geometry;
  };

  struct InputDescTau {
    std::string CollectionName;
    edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken_;
  };

  static constexpr float dR2_max = L2TauTagNNv1::dR_max * L2TauTagNNv1::dR_max;
  static constexpr float dEta_width = 2 * L2TauTagNNv1::dR_max / static_cast<float>(L2TauTagNNv1::nCellEta);
  static constexpr float dPhi_width = 2 * L2TauTagNNv1::dR_max / static_cast<float>(L2TauTagNNv1::nCellPhi);

  explicit L2TauNNProducer(const edm::ParameterSet&, const L2TauNNProducerCacheData*);
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  static std::unique_ptr<L2TauNNProducerCacheData> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(L2TauNNProducerCacheData*);

private:
  void checknan(tensorflow::Tensor& tensor, int debugLevel);
  void standardizeTensor(tensorflow::Tensor& tensor);
  std::vector<float> getTauScore(const tensorflow::Tensor& cellGridMatrix);
  void produce(edm::Event& event, const edm::EventSetup& eventsetup) override;
  void fillL1TauVars(tensorflow::Tensor& cellGridMatrix, const std::vector<l1t::TauRef>& allTaus);
  void fillCaloRecHits(tensorflow::Tensor& cellGridMatrix,
                       const std::vector<l1t::TauRef>& allTaus,
                       const caloRecHitCollections& caloRecHits);
  void fillPatatracks(tensorflow::Tensor& cellGridMatrix,
                      const std::vector<l1t::TauRef>& allTaus,
                      const TrackSoAHost& patatracks_tsoa,
                      const ZVertexSoAHost& patavtx_soa,
                      const reco::BeamSpot& beamspot,
                      const MagneticField* magfi);
  void selectGoodTracksAndVertices(const ZVertexSoAHost& patavtx_soa,
                                   const TrackSoAHost& patatracks_tsoa,
                                   std::vector<int>& trkGood,
                                   std::vector<int>& vtxGood);

  std::pair<float, float> impactParameter(int it,
                                          const TrackSoAHost& patatracks_tsoa,
                                          float patatrackPhi,
                                          const reco::BeamSpot& beamspot,
                                          const MagneticField* magfi);
  template <typename VPos, typename LVec>
  std::tuple<float, float, int, int> getEtaPhiIndices(const VPos& position, const LVec& tau_p4);
  template <typename LVec>
  std::tuple<float, float, int, int> getEtaPhiIndices(float eta, float phi, const LVec& tau_p4);

private:
  const int debugLevel_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tauTriggerToken_;
  std::vector<InputDescTau> L1TauDesc_;
  const edm::EDGetTokenT<HBHERecHitCollection> hbheToken_;
  const edm::EDGetTokenT<HORecHitCollection> hoToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> eeToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bFieldToken_;
  const edm::EDGetTokenT<ZVertexSoAHost> pataVerticesToken_;
  const edm::EDGetTokenT<TrackSoAHost> pataTracksToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const unsigned int maxVtx_;
  const float fractionSumPt2_;
  const float minSumPt2_;
  const float trackPtMin_;
  const float trackPtMax_;
  const float trackChi2Max_;
  std::string inputTensorName_;
  std::string outputTensorName_;
  const L2TauNNProducerCacheData* L2cacheData_;
};

std::unique_ptr<L2TauNNProducerCacheData> L2TauNNProducer::initializeGlobalCache(const edm::ParameterSet& cfg) {
  std::unique_ptr<L2TauNNProducerCacheData> cacheData = std::make_unique<L2TauNNProducerCacheData>();
  cacheData->normVec.reserve(L2TauTagNNv1::nVars);

  auto const graphPath = edm::FileInPath(cfg.getParameter<std::string>("graphPath")).fullPath();

  cacheData->graphDef = tensorflow::loadGraphDef(graphPath);
  cacheData->session = tensorflow::createSession(cacheData->graphDef);

  tensorflow::setLogging("2");

  boost::property_tree::ptree loadPtreeRoot;
  auto const normalizationDict = edm::FileInPath(cfg.getParameter<std::string>("normalizationDict")).fullPath();
  boost::property_tree::read_json(normalizationDict, loadPtreeRoot);
  for (const auto& [key, val] : L2TauTagNNv1::varNameMap) {
    boost::property_tree::ptree var = loadPtreeRoot.get_child(val);
    normDictElement current_element;
    current_element.mean = var.get_child("mean").get_value<float>();
    current_element.std = var.get_child("std").get_value<float>();
    current_element.min = var.get_child("min").get_value<float>();
    current_element.max = var.get_child("max").get_value<float>();
    cacheData->normVec.push_back(current_element);
  }
  return cacheData;
}
void L2TauNNProducer::globalEndJob(L2TauNNProducerCacheData* cacheData) {
  if (cacheData->graphDef != nullptr) {
    delete cacheData->graphDef;
  }
  tensorflow::closeSession(cacheData->session);
}
void L2TauNNProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("debugLevel", 0)->setComment("set debug level for printing out info");
  edm::ParameterSetDescription l1TausPset;
  l1TausPset.add<std::string>("L1CollectionName", "DoubleTau")->setComment("Name of collections");
  l1TausPset.add<edm::InputTag>("L1TauTrigger", edm::InputTag("hltL1sDoubleTauBigOR"))
      ->setComment("Which trigger should the L1 Taus collection pass");
  edm::ParameterSet l1TausPSetDefault;
  l1TausPSetDefault.addParameter<std::string>("L1CollectionName", "DoubleTau");
  l1TausPSetDefault.addParameter<edm::InputTag>("L1TauTrigger", edm::InputTag("hltL1sDoubleTauBigOR"));
  desc.addVPSet("L1Taus", l1TausPset, {l1TausPSetDefault});
  desc.add<edm::InputTag>("hbheInput", edm::InputTag("hltHbhereco"))->setComment("HBHE recHit collection");
  desc.add<edm::InputTag>("hoInput", edm::InputTag("hltHoreco"))->setComment("HO recHit Collection");
  desc.add<edm::InputTag>("ebInput", edm::InputTag("hltEcalRecHit:EcalRecHitsEB"))->setComment("EB recHit Collection");
  desc.add<edm::InputTag>("eeInput", edm::InputTag("hltEcalRecHit:EcalRecHitsEE"))->setComment("EE recHit Collection");
  desc.add<edm::InputTag>("pataVertices", edm::InputTag("hltPixelVerticesSoA"))
      ->setComment("patatrack vertices collection");
  desc.add<edm::InputTag>("pataTracks", edm::InputTag("hltPixelTracksSoA"))->setComment("patatrack collection");
  desc.add<edm::InputTag>("BeamSpot", edm::InputTag("hltOnlineBeamSpot"))->setComment("BeamSpot Collection");
  desc.add<uint>("maxVtx", 100)->setComment("max output collection size (number of accepted vertices)");
  desc.add<double>("fractionSumPt2", 0.3)->setComment("threshold on sumPt2 fraction of the leading vertex");
  desc.add<double>("minSumPt2", 0.)->setComment("min sumPt2");
  desc.add<double>("track_pt_min", 1.0)->setComment("min track p_T");
  desc.add<double>("track_pt_max", 10.0)->setComment("max track p_T");
  desc.add<double>("track_chi2_max", 99999.)->setComment("max track chi2");
  desc.add<std::string>("graphPath", "RecoTauTag/TrainingFiles/data/L2TauNNTag/L2TauTag_Run3v1.pb")
      ->setComment("path to the saved CNN");
  desc.add<std::string>("normalizationDict", "RecoTauTag/TrainingFiles/data/L2TauNNTag/NormalizationDict.json")
      ->setComment("path to the dictionary for variable standardization");
  descriptions.addWithDefaultLabel(desc);
}

L2TauNNProducer::L2TauNNProducer(const edm::ParameterSet& cfg, const L2TauNNProducerCacheData* cacheData)
    : debugLevel_(cfg.getParameter<int>("debugLevel")),
      hbheToken_(consumes<HBHERecHitCollection>(cfg.getParameter<edm::InputTag>("hbheInput"))),
      hoToken_(consumes<HORecHitCollection>(cfg.getParameter<edm::InputTag>("hoInput"))),
      ebToken_(consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("ebInput"))),
      eeToken_(consumes<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("eeInput"))),
      geometryToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      bFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      pataVerticesToken_(consumes(cfg.getParameter<edm::InputTag>("pataVertices"))),
      pataTracksToken_(consumes(cfg.getParameter<edm::InputTag>("pataTracks"))),
      beamSpotToken_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("BeamSpot"))),
      maxVtx_(cfg.getParameter<uint>("maxVtx")),
      fractionSumPt2_(cfg.getParameter<double>("fractionSumPt2")),
      minSumPt2_(cfg.getParameter<double>("minSumPt2")),
      trackPtMin_(cfg.getParameter<double>("track_pt_min")),
      trackPtMax_(cfg.getParameter<double>("track_pt_max")),
      trackChi2Max_(cfg.getParameter<double>("track_chi2_max")) {
  if (cacheData->graphDef == nullptr) {
    throw cms::Exception("InvalidCacheData") << "Invalid Cache Data.";
  }
  inputTensorName_ = cacheData->graphDef->node(0).name();
  outputTensorName_ = cacheData->graphDef->node(cacheData->graphDef->node_size() - 1).name();
  L2cacheData_ = cacheData;
  std::vector<edm::ParameterSet> L1TauCollections = cfg.getParameter<std::vector<edm::ParameterSet>>("L1Taus");
  L1TauDesc_.reserve(L1TauCollections.size());
  for (const auto& l1TauInput : L1TauCollections) {
    InputDescTau toInsert;
    toInsert.CollectionName = l1TauInput.getParameter<std::string>("L1CollectionName");
    toInsert.inputToken_ =
        consumes<trigger::TriggerFilterObjectWithRefs>(l1TauInput.getParameter<edm::InputTag>("L1TauTrigger"));
    L1TauDesc_.push_back(toInsert);
  }
  for (const auto& desc : L1TauDesc_)
    produces<std::vector<float>>(desc.CollectionName);
}

void L2TauNNProducer::checknan(tensorflow::Tensor& tensor, int debugLevel) {
  using NNInputs = L2TauTagNNv1::NNInputs;
  std::vector<int> tensor_shape(tensor.shape().dims());
  for (int d = 0; d < tensor.shape().dims(); d++) {
    tensor_shape.at(d) = tensor.shape().dim_size(d);
  }
  if (tensor_shape.size() != 4) {
    throw cms::Exception("InvalidTensor") << "Tensor shape does not have 4 dimensions!";
  }
  for (int tau_idx = 0; tau_idx < tensor_shape.at(0); tau_idx++) {
    for (int phi_idx = 0; phi_idx < tensor_shape.at(1); phi_idx++) {
      for (int eta_idx = 0; eta_idx < tensor_shape.at(2); eta_idx++) {
        for (int var_idx = 0; var_idx < tensor_shape.at(3); var_idx++) {
          auto getCell = [&](NNInputs input) -> float& {
            return getCellImpl(tensor, tau_idx, phi_idx, eta_idx, input);
          };
          auto nonstd_var = getCell(static_cast<NNInputs>(var_idx));
          if (edm::isNotFinite(nonstd_var)) {
            edm::LogWarning("InputVar") << "var is nan \nvar name= "
                                        << L2TauTagNNv1::varNameMap.at(static_cast<L2TauTagNNv1::NNInputs>(var_idx))
                                        << "\t var_idx = " << var_idx << "\t eta_idx = " << eta_idx
                                        << "\t phi_idx = " << phi_idx << "\t tau_idx = " << tau_idx;
            if (debugLevel > 2) {
              edm::LogWarning("InputVar") << "other vars in same cell \n";
              if (var_idx + 1 < tensor_shape.at(3))
                edm::LogWarning("InputVar") << L2TauTagNNv1::varNameMap.at(static_cast<NNInputs>(var_idx + 1))
                                            << "\t = " << getCell(static_cast<NNInputs>(var_idx + 1));
              if (var_idx + 2 < tensor_shape.at(3))
                edm::LogWarning("InputVar") << L2TauTagNNv1::varNameMap.at(static_cast<NNInputs>(var_idx + 2))
                                            << "\t = " << getCell(static_cast<NNInputs>(var_idx + 2));
              if (var_idx + 3 < tensor_shape.at(3))
                edm::LogWarning("InputVar") << L2TauTagNNv1::varNameMap.at(static_cast<NNInputs>(var_idx + 3))
                                            << "\t = " << getCell(static_cast<NNInputs>(var_idx + 3));
              if (var_idx + 4 < tensor_shape.at(3))
                edm::LogWarning("InputVar") << L2TauTagNNv1::varNameMap.at(static_cast<NNInputs>(var_idx + 4))
                                            << "\t = " << getCell(static_cast<NNInputs>(var_idx + 4));
            }
          }
        }
      }
    }
  }
}

void L2TauNNProducer::standardizeTensor(tensorflow::Tensor& tensor) {
  using NNInputs = L2TauTagNNv1::NNInputs;
  std::vector<int> tensor_shape(tensor.shape().dims());
  for (int d = 0; d < tensor.shape().dims(); d++) {
    tensor_shape.at(d) = tensor.shape().dim_size(d);
  }
  if (tensor_shape.size() != 4) {
    throw cms::Exception("InvalidTensor") << "Tensor shape does not have 4 dimensions!";
  }
  for (int tau_idx = 0; tau_idx < tensor_shape.at(0); tau_idx++) {
    for (int phi_idx = 0; phi_idx < tensor_shape.at(1); phi_idx++) {
      for (int eta_idx = 0; eta_idx < tensor_shape.at(2); eta_idx++) {
        for (int var_idx = 0; var_idx < tensor_shape.at(3); var_idx++) {
          auto getCell = [&](NNInputs input) -> float& {
            return getCellImpl(tensor, tau_idx, phi_idx, eta_idx, input);
          };
          float mean = L2cacheData_->normVec.at(var_idx).mean;
          float std = L2cacheData_->normVec.at(var_idx).std;
          float min = L2cacheData_->normVec.at(var_idx).min;
          float max = L2cacheData_->normVec.at(var_idx).max;
          float nonstd_var = getCell(static_cast<NNInputs>(var_idx));
          float std_var = static_cast<float>((nonstd_var - mean) / std);
          if (std_var > max) {
            std_var = static_cast<float>(max);
          } else if (std_var < min) {
            std_var = static_cast<float>(min);
          }
          getCell(static_cast<NNInputs>(var_idx)) = std_var;
        }
      }
    }
  }
}

void L2TauNNProducer::fillL1TauVars(tensorflow::Tensor& cellGridMatrix, const std::vector<l1t::TauRef>& allTaus) {
  using NNInputs = L2TauTagNNv1::NNInputs;

  const int nTaus = allTaus.size();
  for (int tau_idx = 0; tau_idx < nTaus; tau_idx++) {
    for (int eta_idx = 0; eta_idx < L2TauTagNNv1::nCellEta; eta_idx++) {
      for (int phi_idx = 0; phi_idx < L2TauTagNNv1::nCellPhi; phi_idx++) {
        auto getCell = [&](NNInputs input) -> float& {
          return getCellImpl(cellGridMatrix, tau_idx, phi_idx, eta_idx, input);
        };
        getCell(NNInputs::l1Tau_pt) = allTaus[tau_idx]->pt();
        getCell(NNInputs::l1Tau_eta) = allTaus[tau_idx]->eta();
        getCell(NNInputs::l1Tau_hwIso) = allTaus[tau_idx]->hwIso();
      }
    }
  }
}

template <typename LVec>
std::tuple<float, float, int, int> L2TauNNProducer::getEtaPhiIndices(float eta, float phi, const LVec& tau_p4) {
  const float deta = eta - tau_p4.eta();
  const float dphi = reco::deltaPhi(phi, tau_p4.phi());
  const int eta_idx = static_cast<int>(floor((deta + L2TauTagNNv1::dR_max) / dEta_width));
  const int phi_idx = static_cast<int>(floor((dphi + L2TauTagNNv1::dR_max) / dPhi_width));
  return std::make_tuple(deta, dphi, eta_idx, phi_idx);
}

template <typename VPos, typename LVec>
std::tuple<float, float, int, int> L2TauNNProducer::getEtaPhiIndices(const VPos& position, const LVec& tau_p4) {
  return getEtaPhiIndices(position.eta(), position.phi(), tau_p4);
}

void L2TauNNProducer::fillCaloRecHits(tensorflow::Tensor& cellGridMatrix,
                                      const std::vector<l1t::TauRef>& allTaus,
                                      const caloRecHitCollections& caloRecHits) {
  using NNInputs = L2TauTagNNv1::NNInputs;

  const int nTaus = allTaus.size();
  float deta, dphi;
  int eta_idx = 0;
  int phi_idx = 0;
  int tau_idx = 0;

  auto getCell = [&](NNInputs input) -> float& {
    return getCellImpl(cellGridMatrix, tau_idx, phi_idx, eta_idx, input);
  };
  for (tau_idx = 0; tau_idx < nTaus; tau_idx++) {
    // calorechit_EE
    for (const auto& caloRecHit_ee : *caloRecHits.ee) {
      if (caloRecHit_ee.energy() <= 0)
        continue;
      const auto& position = caloRecHits.geometry->getGeometry(caloRecHit_ee.id())->getPosition();
      const float eeCalEn = caloRecHit_ee.energy();
      const float eeCalChi2 = caloRecHit_ee.chi2();
      if (reco::deltaR2(position, allTaus[tau_idx]->polarP4()) < dR2_max) {
        std::tie(deta, dphi, eta_idx, phi_idx) = getEtaPhiIndices(position, allTaus[tau_idx]->polarP4());
        getCell(NNInputs::EcalEnergySum) += eeCalEn;
        getCell(NNInputs::EcalSize) += 1.;
        getCell(NNInputs::EcalEnergyStdDev) += eeCalEn * eeCalEn;
        getCell(NNInputs::EcalDeltaEta) += deta * eeCalEn;
        getCell(NNInputs::EcalDeltaPhi) += dphi * eeCalEn;
        if (eeCalChi2 >= 0) {
          getCell(NNInputs::EcalChi2) += eeCalChi2 * eeCalEn;
          getCell(NNInputs::EcalEnergySumForPositiveChi2) += eeCalEn;
          getCell(NNInputs::EcalSizeForPositiveChi2) += 1.;
        }
      }
    }

    // calorechit_EB
    for (const auto& caloRecHit_eb : *caloRecHits.eb) {
      if (caloRecHit_eb.energy() <= 0)
        continue;
      const auto& position = caloRecHits.geometry->getGeometry(caloRecHit_eb.id())->getPosition();
      const float ebCalEn = caloRecHit_eb.energy();
      const float ebCalChi2 = caloRecHit_eb.chi2();
      if (reco::deltaR2(position, allTaus[tau_idx]->polarP4()) < dR2_max) {
        std::tie(deta, dphi, eta_idx, phi_idx) = getEtaPhiIndices(position, allTaus[tau_idx]->polarP4());
        getCell(NNInputs::EcalEnergySum) += ebCalEn;
        getCell(NNInputs::EcalSize) += 1.;
        getCell(NNInputs::EcalEnergyStdDev) += ebCalEn * ebCalEn;
        getCell(NNInputs::EcalDeltaEta) += deta * ebCalEn;
        getCell(NNInputs::EcalDeltaPhi) += dphi * ebCalEn;
        if (ebCalChi2 >= 0) {
          getCell(NNInputs::EcalChi2) += ebCalChi2 * ebCalEn;
          getCell(NNInputs::EcalEnergySumForPositiveChi2) += ebCalEn;
          getCell(NNInputs::EcalSizeForPositiveChi2) += 1.;
        }
      }
    }

    // calorechit_HBHE
    for (const auto& caloRecHit_hbhe : *caloRecHits.hbhe) {
      if (caloRecHit_hbhe.energy() <= 0)
        continue;
      const auto& position = caloRecHits.geometry->getGeometry(caloRecHit_hbhe.id())->getPosition();
      const float hbheCalEn = caloRecHit_hbhe.energy();
      const float hbheCalChi2 = caloRecHit_hbhe.chi2();
      if (reco::deltaR2(position, allTaus[tau_idx]->polarP4()) < dR2_max) {
        std::tie(deta, dphi, eta_idx, phi_idx) = getEtaPhiIndices(position, allTaus[tau_idx]->polarP4());
        getCell(NNInputs::HcalEnergySum) += hbheCalEn;
        getCell(NNInputs::HcalEnergyStdDev) += hbheCalEn * hbheCalEn;
        getCell(NNInputs::HcalSize) += 1.;
        getCell(NNInputs::HcalDeltaEta) += deta * hbheCalEn;
        getCell(NNInputs::HcalDeltaPhi) += dphi * hbheCalEn;
        if (hbheCalChi2 >= 0) {
          getCell(NNInputs::HcalChi2) += hbheCalChi2 * hbheCalEn;
          getCell(NNInputs::HcalEnergySumForPositiveChi2) += hbheCalEn;
          getCell(NNInputs::HcalSizeForPositiveChi2) += 1.;
        }
      }
    }

    // calorechit_HO
    for (const auto& caloRecHit_ho : *caloRecHits.ho) {
      if (caloRecHit_ho.energy() <= 0)
        continue;
      const auto& position = caloRecHits.geometry->getGeometry(caloRecHit_ho.id())->getPosition();
      const float hoCalEn = caloRecHit_ho.energy();
      if (reco::deltaR2(position, allTaus[tau_idx]->polarP4()) < dR2_max) {
        std::tie(deta, dphi, eta_idx, phi_idx) = getEtaPhiIndices(position, allTaus[tau_idx]->polarP4());
        getCell(NNInputs::HcalEnergySum) += hoCalEn;
        getCell(NNInputs::HcalEnergyStdDev) += hoCalEn * hoCalEn;
        getCell(NNInputs::HcalSize) += 1.;
        getCell(NNInputs::HcalDeltaEta) += deta * hoCalEn;
        getCell(NNInputs::HcalDeltaPhi) += dphi * hoCalEn;
      }
    }

    // normalize to sum and define stdDev
    for (eta_idx = 0; eta_idx < L2TauTagNNv1::nCellEta; eta_idx++) {
      for (phi_idx = 0; phi_idx < L2TauTagNNv1::nCellPhi; phi_idx++) {
        /* normalize eCal vars*/
        if (getCell(NNInputs::EcalEnergySum) > 0.) {
          getCell(NNInputs::EcalDeltaEta) /= getCell(NNInputs::EcalEnergySum);
          getCell(NNInputs::EcalDeltaPhi) /= getCell(NNInputs::EcalEnergySum);
        }
        if (getCell(NNInputs::EcalEnergySumForPositiveChi2) > 0.) {
          getCell(NNInputs::EcalChi2) /= getCell(NNInputs::EcalEnergySumForPositiveChi2);
        }
        if (getCell(NNInputs::EcalSize) > 1.) {
          // (stdDev - (enSum*enSum)/size) / (size-1)
          getCell(NNInputs::EcalEnergyStdDev) =
              (getCell(NNInputs::EcalEnergyStdDev) -
               (getCell(NNInputs::EcalEnergySum) * getCell(NNInputs::EcalEnergySum)) / getCell(NNInputs::EcalSize)) /
              (getCell(NNInputs::EcalSize) - 1);
        } else {
          getCell(NNInputs::EcalEnergyStdDev) = 0.;
        }
        /* normalize hCal Vars */
        if (getCell(NNInputs::HcalEnergySum) > 0.) {
          getCell(NNInputs::HcalDeltaEta) /= getCell(NNInputs::HcalEnergySum);
          getCell(NNInputs::HcalDeltaPhi) /= getCell(NNInputs::HcalEnergySum);
        }
        if (getCell(NNInputs::HcalEnergySumForPositiveChi2) > 0.) {
          getCell(NNInputs::HcalChi2) /= getCell(NNInputs::HcalEnergySumForPositiveChi2);
        }
        if (getCell(NNInputs::HcalSize) > 1.) {
          // (stdDev - (enSum*enSum)/size) / (size-1)
          getCell(NNInputs::HcalEnergyStdDev) =
              (getCell(NNInputs::HcalEnergyStdDev) -
               (getCell(NNInputs::HcalEnergySum) * getCell(NNInputs::HcalEnergySum)) / getCell(NNInputs::HcalSize)) /
              (getCell(NNInputs::HcalSize) - 1);
        } else {
          getCell(NNInputs::HcalEnergyStdDev) = 0.;
        }
      }
    }
  }
}

void L2TauNNProducer::selectGoodTracksAndVertices(const ZVertexSoAHost& patavtx_soa,
                                                  const TrackSoAHost& patatracks_tsoa,
                                                  std::vector<int>& trkGood,
                                                  std::vector<int>& vtxGood) {
  using patatrackHelpers = TracksUtilities<pixelTopology::Phase1>;
  const auto maxTracks = patatracks_tsoa.view().metadata().size();
  const int nv = patavtx_soa.view().nvFinal();
  trkGood.clear();
  trkGood.reserve(maxTracks);
  vtxGood.clear();
  vtxGood.reserve(nv);
  auto const* quality = patatracks_tsoa.view().quality();

  // No need to sort either as the algorithms is just using the max (not even the location, just the max value of pt2sum).
  std::vector<float> pTSquaredSum(nv, 0);
  std::vector<int> nTrkAssociated(nv, 0);

  for (int32_t trk_idx = 0; trk_idx < maxTracks; ++trk_idx) {
    auto nHits = patatrackHelpers::nHits(patatracks_tsoa.view(), trk_idx);
    if (nHits == 0) {
      break;
    }
    int vtx_ass_to_track = patavtx_soa.view()[trk_idx].idv();
    if (vtx_ass_to_track >= 0 && vtx_ass_to_track < nv) {
      auto patatrackPt = patatracks_tsoa.view()[trk_idx].pt();
      ++nTrkAssociated[vtx_ass_to_track];
      if (patatrackPt >= trackPtMin_ && patatracks_tsoa.const_view()[trk_idx].chi2() <= trackChi2Max_) {
        patatrackPt = std::min(patatrackPt, trackPtMax_);
        pTSquaredSum[vtx_ass_to_track] += patatrackPt * patatrackPt;
      }
    }
    if (nHits > 0 and quality[trk_idx] >= pixelTrack::Quality::loose) {
      trkGood.push_back(trk_idx);
    }
  }
  if (nv > 0) {
    const auto minFOM_fromFrac = (*std::max_element(pTSquaredSum.begin(), pTSquaredSum.end())) * fractionSumPt2_;
    for (int j = nv - 1; j >= 0 && vtxGood.size() < maxVtx_; --j) {
      auto vtx_idx = patavtx_soa.view()[j].sortInd();
      assert(vtx_idx < nv);
      if (nTrkAssociated[vtx_idx] >= 2 && pTSquaredSum[vtx_idx] >= minFOM_fromFrac &&
          pTSquaredSum[vtx_idx] > minSumPt2_) {
        vtxGood.push_back(vtx_idx);
      }
    }
  }
}

std::pair<float, float> L2TauNNProducer::impactParameter(int it,
                                                         const TrackSoAHost& patatracks_tsoa,
                                                         float patatrackPhi,
                                                         const reco::BeamSpot& beamspot,
                                                         const MagneticField* magfi) {
  /* dxy and dz */
  riemannFit::Vector5d ipar, opar;
  riemannFit::Matrix5d icov, ocov;
  TracksUtilities<pixelTopology::Phase1>::copyToDense(patatracks_tsoa.view(), ipar, icov, it);
  riemannFit::transformToPerigeePlane(ipar, icov, opar, ocov);
  LocalTrajectoryParameters lpar(opar(0), opar(1), opar(2), opar(3), opar(4), 1.);
  float sp = std::sin(patatrackPhi);
  float cp = std::cos(patatrackPhi);
  Surface::RotationType Rotation(sp, -cp, 0, 0, 0, -1.f, cp, sp, 0);
  GlobalPoint BeamSpotPoint(beamspot.x0(), beamspot.y0(), beamspot.z0());
  Plane impPointPlane(BeamSpotPoint, Rotation);
  GlobalTrajectoryParameters gp(
      impPointPlane.toGlobal(lpar.position()), impPointPlane.toGlobal(lpar.momentum()), lpar.charge(), magfi);
  GlobalPoint vv = gp.position();
  math::XYZPoint pos(vv.x(), vv.y(), vv.z());
  GlobalVector pp = gp.momentum();
  math::XYZVector mom(pp.x(), pp.y(), pp.z());
  auto lambda = M_PI_2 - pp.theta();
  auto phi = pp.phi();
  float patatrackDxy = -vv.x() * std::sin(phi) + vv.y() * std::cos(phi);
  float patatrackDz =
      (vv.z() * std::cos(lambda) - (vv.x() * std::cos(phi) + vv.y() * std::sin(phi)) * std::sin(lambda)) /
      std::cos(lambda);
  return std::make_pair(patatrackDxy, patatrackDz);
}

void L2TauNNProducer::fillPatatracks(tensorflow::Tensor& cellGridMatrix,
                                     const std::vector<l1t::TauRef>& allTaus,
                                     const TrackSoAHost& patatracks_tsoa,
                                     const ZVertexSoAHost& patavtx_soa,
                                     const reco::BeamSpot& beamspot,
                                     const MagneticField* magfi) {
  using NNInputs = L2TauTagNNv1::NNInputs;
  using patatrackHelpers = TracksUtilities<pixelTopology::Phase1>;
  float deta, dphi;
  int eta_idx = 0;
  int phi_idx = 0;
  int tau_idx = 0;

  auto getCell = [&](NNInputs input) -> float& {
    return getCellImpl(cellGridMatrix, tau_idx, phi_idx, eta_idx, input);
  };

  std::vector<int> trkGood;
  std::vector<int> vtxGood;

  selectGoodTracksAndVertices(patavtx_soa, patatracks_tsoa, trkGood, vtxGood);

  const int nTaus = allTaus.size();
  for (tau_idx = 0; tau_idx < nTaus; tau_idx++) {
    const float tauEta = allTaus[tau_idx]->eta();
    const float tauPhi = allTaus[tau_idx]->phi();

    for (const auto it : trkGood) {
      const float patatrackPt = patatracks_tsoa.const_view()[it].pt();
      if (patatrackPt <= 0)
        continue;
      const float patatrackPhi = patatrackHelpers::phi(patatracks_tsoa.const_view(), it);
      const float patatrackEta = patatracks_tsoa.const_view()[it].eta();
      const float patatrackCharge = patatrackHelpers::charge(patatracks_tsoa.const_view(), it);
      const float patatrackChi2OverNdof = patatracks_tsoa.view()[it].chi2();
      const auto nHits = patatrackHelpers::nHits(patatracks_tsoa.const_view(), it);
      if (nHits <= 0)
        continue;
      const int patatrackNdof = 2 * std::min(6, nHits) - 5;

      const int vtx_idx_assTrk = patavtx_soa.view()[it].idv();
      if (reco::deltaR2(patatrackEta, patatrackPhi, tauEta, tauPhi) < dR2_max) {
        std::tie(deta, dphi, eta_idx, phi_idx) =
            getEtaPhiIndices(patatrackEta, patatrackPhi, allTaus[tau_idx]->polarP4());
        getCell(NNInputs::PatatrackPtSum) += patatrackPt;
        getCell(NNInputs::PatatrackSize) += 1.;
        getCell(NNInputs::PatatrackChargeSum) += patatrackCharge;
        getCell(NNInputs::PatatrackDeltaEta) += deta * patatrackPt;
        getCell(NNInputs::PatatrackDeltaPhi) += dphi * patatrackPt;
        getCell(NNInputs::PatatrackChi2OverNdof) += patatrackChi2OverNdof * patatrackPt;
        getCell(NNInputs::PatatrackNdof) += patatrackNdof * patatrackPt;
        std::pair<float, float> impactParameters = impactParameter(it, patatracks_tsoa, patatrackPhi, beamspot, magfi);
        getCell(NNInputs::PatatrackDxy) += impactParameters.first * patatrackPt;
        getCell(NNInputs::PatatrackDz) += impactParameters.second * patatrackPt;
        if ((std::find(vtxGood.begin(), vtxGood.end(), vtx_idx_assTrk) != vtxGood.end())) {
          getCell(NNInputs::PatatrackPtSumWithVertex) += patatrackPt;
          getCell(NNInputs::PatatrackSizeWithVertex) += 1.;
        }
      }
    }

    // normalize to sum and define stdDev
    for (eta_idx = 0; eta_idx < L2TauTagNNv1::nCellEta; eta_idx++) {
      for (phi_idx = 0; phi_idx < L2TauTagNNv1::nCellPhi; phi_idx++) {
        getCell(NNInputs::nVertices) = vtxGood.size();
        if (getCell(NNInputs::PatatrackPtSum) > 0.) {
          getCell(NNInputs::PatatrackDeltaEta) /= getCell(NNInputs::PatatrackPtSum);
          getCell(NNInputs::PatatrackDeltaPhi) /= getCell(NNInputs::PatatrackPtSum);
          getCell(NNInputs::PatatrackChi2OverNdof) /= getCell(NNInputs::PatatrackPtSum);
          getCell(NNInputs::PatatrackNdof) /= getCell(NNInputs::PatatrackPtSum);
          getCell(NNInputs::PatatrackDxy) /= getCell(NNInputs::PatatrackPtSum);
          getCell(NNInputs::PatatrackDz) /= getCell(NNInputs::PatatrackPtSum);
        }
      }
    }
  }
}

std::vector<float> L2TauNNProducer::getTauScore(const tensorflow::Tensor& cellGridMatrix) {
  std::vector<tensorflow::Tensor> pred_tensor;
  tensorflow::run(L2cacheData_->session, {{inputTensorName_, cellGridMatrix}}, {outputTensorName_}, &pred_tensor);
  const int nTau = cellGridMatrix.shape().dim_size(0);
  std::vector<float> pred_vector(nTau);
  for (int tau_idx = 0; tau_idx < nTau; ++tau_idx) {
    pred_vector[tau_idx] = pred_tensor[0].matrix<float>()(tau_idx, 0);
  }

  return pred_vector;
}

void L2TauNNProducer::produce(edm::Event& event, const edm::EventSetup& eventsetup) {
  std::vector<std::vector<size_t>> TauCollectionMap(L1TauDesc_.size());
  l1t::TauVectorRef allTaus;

  for (size_t inp_idx = 0; inp_idx < L1TauDesc_.size(); inp_idx++) {
    l1t::TauVectorRef l1Taus;
    auto const& l1TriggeredTaus = event.get(L1TauDesc_[inp_idx].inputToken_);
    l1TriggeredTaus.getObjects(trigger::TriggerL1Tau, l1Taus);
    TauCollectionMap.at(inp_idx).resize(l1Taus.size());

    for (size_t l1_idx = 0; l1_idx < l1Taus.size(); l1_idx++) {
      size_t tau_idx;
      const auto iter = std::find(allTaus.begin(), allTaus.end(), l1Taus[l1_idx]);
      if (iter != allTaus.end()) {
        tau_idx = std::distance(allTaus.begin(), iter);
      } else {
        allTaus.push_back(l1Taus[l1_idx]);
        tau_idx = allTaus.size() - 1;
      }
      TauCollectionMap.at(inp_idx).at(l1_idx) = tau_idx;
    }
  }
  const auto ebCal = event.getHandle(ebToken_);
  const auto eeCal = event.getHandle(eeToken_);
  const auto hbhe = event.getHandle(hbheToken_);
  const auto ho = event.getHandle(hoToken_);
  auto const& patatracks_SoA = event.get(pataTracksToken_);
  auto const& vertices_SoA = event.get(pataVerticesToken_);
  const auto bsHandle = event.getHandle(beamSpotToken_);

  auto const fieldESH = eventsetup.getHandle(bFieldToken_);
  auto const geometry = eventsetup.getHandle(geometryToken_);

  caloRecHitCollections caloRecHits;
  caloRecHits.hbhe = &*hbhe;
  caloRecHits.ho = &*ho;
  caloRecHits.eb = &*ebCal;
  caloRecHits.ee = &*eeCal;
  caloRecHits.geometry = &*geometry;

  const int nTaus = allTaus.size();
  tensorflow::Tensor cellGridMatrix(tensorflow::DT_FLOAT,
                                    {nTaus, L2TauTagNNv1::nCellEta, L2TauTagNNv1::nCellPhi, L2TauTagNNv1::nVars});
  const int n_inputs = nTaus * L2TauTagNNv1::nCellEta * L2TauTagNNv1::nCellPhi * L2TauTagNNv1::nVars;
  for (int input_idx = 0; input_idx < n_inputs; ++input_idx) {
    cellGridMatrix.flat<float>()(input_idx) = 0;
  }
  fillL1TauVars(cellGridMatrix, allTaus);

  fillCaloRecHits(cellGridMatrix, allTaus, caloRecHits);

  fillPatatracks(cellGridMatrix, allTaus, patatracks_SoA, vertices_SoA, *bsHandle, fieldESH.product());

  standardizeTensor(cellGridMatrix);

  if (debugLevel_ > 0) {
    checknan(cellGridMatrix, debugLevel_);
  }

  std::vector<float> tau_score = getTauScore(cellGridMatrix);

  for (size_t inp_idx = 0; inp_idx < L1TauDesc_.size(); inp_idx++) {
    const size_t nTau = TauCollectionMap[inp_idx].size();
    auto tau_tags = std::make_unique<std::vector<float>>(nTau);
    for (size_t tau_pos = 0; tau_pos < nTau; ++tau_pos) {
      const auto tau_idx = TauCollectionMap[inp_idx][tau_pos];
      if (debugLevel_ > 0) {
        edm::LogInfo("DebugInfo") << event.id().event() << " \t " << (allTaus[tau_idx])->pt() << " \t "
                                  << tau_score.at(tau_idx) << std::endl;
      }
      (*tau_tags)[tau_pos] = tau_score.at(tau_idx);
    }
    event.put(std::move(tau_tags), L1TauDesc_[inp_idx].CollectionName);
  }
}
//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L2TauNNProducer);
