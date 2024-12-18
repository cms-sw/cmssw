#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonXGBoostEstimator.h"

#include <memory>
#include <vector>

class PhotonXGBoostProducer : public edm::global::EDProducer<> {
public:
  explicit PhotonXGBoostProducer(edm::ParameterSet const&);
  ~PhotonXGBoostProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> candToken_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> tokenR9_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> tokenHoE_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> tokenSigmaiEtaiEta_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> tokenE2x2_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> tokenIso_;
  const edm::FileInPath mvaFileXgbB_;
  const edm::FileInPath mvaFileXgbE_;
  const unsigned mvaNTreeLimitB_;
  const unsigned mvaNTreeLimitE_;
  const double mvaThresholdEt_;
  const std::unique_ptr<const PhotonXGBoostEstimator> mvaEstimatorB_;
  const std::unique_ptr<const PhotonXGBoostEstimator> mvaEstimatorE_;
};

PhotonXGBoostProducer::PhotonXGBoostProducer(edm::ParameterSet const& config)
    : candToken_(consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("candTag"))),
      tokenR9_(consumes<reco::RecoEcalCandidateIsolationMap>(config.getParameter<edm::InputTag>("inputTagR9"))),
      tokenHoE_(consumes<reco::RecoEcalCandidateIsolationMap>(config.getParameter<edm::InputTag>("inputTagHoE"))),
      tokenSigmaiEtaiEta_(
          consumes<reco::RecoEcalCandidateIsolationMap>(config.getParameter<edm::InputTag>("inputTagSigmaiEtaiEta"))),
      tokenE2x2_(consumes<reco::RecoEcalCandidateIsolationMap>(config.getParameter<edm::InputTag>("inputTagE2x2"))),
      tokenIso_(consumes<reco::RecoEcalCandidateIsolationMap>(config.getParameter<edm::InputTag>("inputTagIso"))),
      mvaFileXgbB_(config.getParameter<edm::FileInPath>("mvaFileXgbB")),
      mvaFileXgbE_(config.getParameter<edm::FileInPath>("mvaFileXgbE")),
      mvaNTreeLimitB_(config.getParameter<unsigned int>("mvaNTreeLimitB")),
      mvaNTreeLimitE_(config.getParameter<unsigned int>("mvaNTreeLimitE")),
      mvaThresholdEt_(config.getParameter<double>("mvaThresholdEt")),
      mvaEstimatorB_{std::make_unique<const PhotonXGBoostEstimator>(mvaFileXgbB_, mvaNTreeLimitB_)},
      mvaEstimatorE_{std::make_unique<const PhotonXGBoostEstimator>(mvaFileXgbE_, mvaNTreeLimitE_)} {
  produces<reco::RecoEcalCandidateIsolationMap>();
}

void PhotonXGBoostProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltEgammaCandidatesUnseeded"));
  desc.add<edm::InputTag>("inputTagR9", edm::InputTag("hltEgammaR9IDUnseeded", "r95x5"));
  desc.add<edm::InputTag>("inputTagHoE", edm::InputTag("hltEgammaHoverEUnseeded"));
  desc.add<edm::InputTag>("inputTagSigmaiEtaiEta",
                          edm::InputTag("hltEgammaClusterShapeUnseeded", "sigmaIEtaIEta5x5NoiseCleaned"));
  desc.add<edm::InputTag>("inputTagE2x2", edm::InputTag("hltEgammaClusterShapeUnseeded", "e2x2"));
  desc.add<edm::InputTag>("inputTagIso", edm::InputTag("hltEgammaEcalPFClusterIsoUnseeded"));
  desc.add<edm::FileInPath>(
      "mvaFileXgbB", edm::FileInPath("RecoEgamma/PhotonIdentification/data/XGBoost/Photon_NTL_168_Barrel_v1.bin"));
  desc.add<edm::FileInPath>(
      "mvaFileXgbE", edm::FileInPath("RecoEgamma/PhotonIdentification/data/XGBoost/Photon_NTL_158_Endcap_v1.bin"));
  desc.add<unsigned int>("mvaNTreeLimitB", 168);
  desc.add<unsigned int>("mvaNTreeLimitE", 158);
  desc.add<double>("mvaThresholdEt", 0);
  descriptions.addWithDefaultLabel(desc);
}

void PhotonXGBoostProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const& setup) const {
  const auto& recCollection = event.getHandle(candToken_);

  //get hold of r9 association map
  const auto& r9Map = event.getHandle(tokenR9_);

  //get hold of HoE association map
  const auto& hoEMap = event.getHandle(tokenHoE_);

  //get hold of isolated association map
  const auto& sigmaiEtaiEtaMap = event.getHandle(tokenSigmaiEtaiEta_);

  //get hold of e2x2 (s4) association map
  const auto& e2x2Map = event.getHandle(tokenE2x2_);

  //get hold of Ecal isolation association map
  const auto& isoMap = event.getHandle(tokenIso_);

  //output
  reco::RecoEcalCandidateIsolationMap mvaScoreMap(recCollection);

  for (size_t i = 0; i < recCollection->size(); i++) {
    edm::Ref<reco::RecoEcalCandidateCollection> ref(recCollection, i);

    float etaSC = ref->eta();

    float scEnergy = ref->superCluster()->energy();
    float r9 = (*r9Map).find(ref)->val;
    float hoe = (*hoEMap).find(ref)->val / scEnergy;
    float siEtaiEta = (*sigmaiEtaiEtaMap).find(ref)->val;
    float e2x2 = (*e2x2Map).find(ref)->val;
    float s4 = e2x2 / scEnergy;
    float iso = (*isoMap).find(ref)->val;

    float rawEnergy = ref->superCluster()->rawEnergy();
    float etaW = ref->superCluster()->etaWidth();
    float phiW = ref->superCluster()->phiWidth();

    float scEt = scEnergy / cosh(etaSC);
    if (scEt < 0.)
      scEt = 0.; /* first and second order terms assume non-negative energies */

    float xgbScore = -100.;
    //compute only above threshold used for training and cand filter, else store negative value into the association map.
    if (scEt >= mvaThresholdEt_) {
      if (std::abs(etaSC) < 1.5)
        xgbScore = mvaEstimatorB_->computeMva(rawEnergy, r9, siEtaiEta, etaW, phiW, s4, etaSC, hoe, iso);
      else
        xgbScore = mvaEstimatorE_->computeMva(rawEnergy, r9, siEtaiEta, etaW, phiW, s4, etaSC, hoe, iso);
    }
    mvaScoreMap.insert(ref, xgbScore);
  }
  event.put(std::make_unique<reco::RecoEcalCandidateIsolationMap>(mvaScoreMap));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PhotonXGBoostProducer);
