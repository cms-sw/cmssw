#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "RecoJets/JetProducers/interface/QGTagger.h"
#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"

/**
 * EDProducer class to produced the qgLikelihood values and related variables
 * If the input jets are uncorrected, the jecService should be provided, so jet are corrected on the fly before the algorithm is applied
 * Authors: andrea.carlo.marini@cern.ch, tom.cornelis@cern.ch, cms-qg-workinggroup@cern.ch
 */
QGTagger::QGTagger(const edm::ParameterSet& iConfig)
    : jetsToken(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("srcJets"))),
      jetCorrectorToken(consumes<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("jec"))),
      vertexToken(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("srcVertexCollection"))),
      rhoToken(consumes<double>(iConfig.getParameter<edm::InputTag>("srcRho"))),
      computeLikelihood(iConfig.getParameter<bool>("computeLikelihood")),
      paramsToken(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("jetsLabel")))),
      useQC(iConfig.getParameter<bool>("useQualityCuts")),
      useJetCorr(!iConfig.getParameter<edm::InputTag>("jec").label().empty()),
      produceSyst(!iConfig.getParameter<std::string>("systematicsLabel").empty()),
      applyConstituentWeight(false) {
  produces<edm::ValueMap<float>>("axis2");
  produces<edm::ValueMap<int>>("mult");
  produces<edm::ValueMap<float>>("ptD");
  if (computeLikelihood) {
    produces<edm::ValueMap<float>>("qgLikelihood");
  }

  edm::InputTag srcConstituentWeights = iConfig.getParameter<edm::InputTag>("srcConstituentWeights");
  if (!srcConstituentWeights.label().empty()) {
    constituentWeightsToken = consumes<edm::ValueMap<float>>(srcConstituentWeights);
    applyConstituentWeight = true;
  }

  if (computeLikelihood && produceSyst) {
    systToken = esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("systematicsLabel")));
    produces<edm::ValueMap<float>>("qgLikelihoodSmearedQuark");
    produces<edm::ValueMap<float>>("qgLikelihoodSmearedGluon");
    produces<edm::ValueMap<float>>("qgLikelihoodSmearedAll");
  }
}

/// Produce qgLikelihood using {mult, ptD, -log(axis2)}
void QGTagger::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  std::vector<float> qgProduct;
  std::vector<float> axis2Product;
  std::vector<int> multProduct;
  std::vector<float> ptDProduct;
  std::vector<float> smearedQuarkProduct;
  std::vector<float> smearedGluonProduct;
  std::vector<float> smearedAllProduct;

  edm::Handle<edm::View<reco::Jet>> jets;
  iEvent.getByToken(jetsToken, jets);
  edm::Handle<reco::JetCorrector> jetCorr;
  if (useJetCorr)
    iEvent.getByToken(jetCorrectorToken, jetCorr);
  edm::Handle<reco::VertexCollection> vertexCollection;
  iEvent.getByToken(vertexToken, vertexCollection);
  edm::Handle<double> rho;
  iEvent.getByToken(rhoToken, rho);

  edm::ValueMap<float> constituentWeights;
  if (applyConstituentWeight) {
    constituentWeights = iEvent.get(constituentWeightsToken);
  }

  const QGLikelihoodObject* QGLParamsColl = nullptr;
  if (computeLikelihood) {
    QGLParamsColl = &iSetup.getData(paramsToken);
  }

  const QGLikelihoodSystematicsObject* QGLSystColl = nullptr;
  if (produceSyst) {
    QGLSystColl = &iSetup.getData(systToken);
  }

  bool weStillNeedToCheckJetCandidates = true;
  bool weAreUsingPackedCandidates = false;
  for (auto jet = jets->begin(); jet != jets->end(); ++jet) {
    if (weStillNeedToCheckJetCandidates) {
      weAreUsingPackedCandidates = isPackedCandidate(&*jet);
      weStillNeedToCheckJetCandidates = false;
    }
    float pt = (useJetCorr ? jet->pt() * jetCorr->correction(*jet) : jet->pt());

    float ptD, axis2;
    int mult;
    std::tie(mult, ptD, axis2) = calcVariables(&*jet, vertexCollection, constituentWeights, weAreUsingPackedCandidates);

    float qgValue;
    if (mult > 2 && computeLikelihood)
      qgValue =
          qgLikelihood.computeQGLikelihood(*QGLParamsColl, pt, jet->eta(), *rho, {(float)mult, ptD, -std::log(axis2)});
    else
      qgValue = -1;

    qgProduct.push_back(qgValue);
    if (computeLikelihood && produceSyst) {
      smearedQuarkProduct.push_back(qgLikelihood.systematicSmearing(*QGLSystColl, pt, jet->eta(), *rho, qgValue, 0));
      smearedGluonProduct.push_back(qgLikelihood.systematicSmearing(*QGLSystColl, pt, jet->eta(), *rho, qgValue, 1));
      smearedAllProduct.push_back(qgLikelihood.systematicSmearing(*QGLSystColl, pt, jet->eta(), *rho, qgValue, 2));
    }
    axis2Product.push_back(axis2);
    multProduct.push_back(mult);
    ptDProduct.push_back(ptD);
  }

  putInEvent("axis2", jets, axis2Product, iEvent);
  putInEvent("mult", jets, multProduct, iEvent);
  putInEvent("ptD", jets, ptDProduct, iEvent);
  if (computeLikelihood) {
    putInEvent("qgLikelihood", jets, qgProduct, iEvent);
    if (produceSyst) {
      putInEvent("qgLikelihoodSmearedQuark", jets, smearedQuarkProduct, iEvent);
      putInEvent("qgLikelihoodSmearedGluon", jets, smearedGluonProduct, iEvent);
      putInEvent("qgLikelihoodSmearedAll", jets, smearedAllProduct, iEvent);
    }
  }
}

/// Function to put product into event
template <typename T>
void QGTagger::putInEvent(const std::string& name,
                          const edm::Handle<edm::View<reco::Jet>>& jets,
                          const std::vector<T>& product,
                          edm::Event& iEvent) const {
  auto out = std::make_unique<edm::ValueMap<T>>();
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(jets, product.begin(), product.end());
  filler.fill();
  iEvent.put(std::move(out), name);
}

/// Function to tell us if we are using packedCandidates, only test for first candidate
bool QGTagger::isPackedCandidate(const reco::Jet* jet) const {
  for (auto candidate : jet->getJetConstituentsQuick()) {
    if (typeid(pat::PackedCandidate) == typeid(*candidate))
      return true;
    else if (typeid(reco::PFCandidate) == typeid(*candidate))
      return false;
    else
      throw cms::Exception("WrongJetCollection", "Jet constituents are not particle flow candidates");
  }
  return false;
}

/// Calculation of axis2, mult and ptD
std::tuple<int, float, float> QGTagger::calcVariables(const reco::Jet* jet,
                                                      edm::Handle<reco::VertexCollection>& vC,
                                                      edm::ValueMap<float>& constituentWeights,
                                                      bool weAreUsingPackedCandidates) const {
  float sum_weight = 0., sum_deta = 0., sum_dphi = 0., sum_deta2 = 0., sum_dphi2 = 0., sum_detadphi = 0., sum_pt = 0.;

  float multWeighted = 0;

  //Loop over the jet constituents
  for (const auto& daughter : jet->getJetConstituents()) {
    const reco::Candidate* cand = daughter.get();

    float constWeight = 1.0;
    if (applyConstituentWeight) {
      constWeight = constituentWeights[daughter];
    }

    if (weAreUsingPackedCandidates) {  //packed candidate situation
      auto part = static_cast<const pat::PackedCandidate*>(cand);

      if (part->charge()) {
        if (!(part->fromPV() > 1 && part->trackHighPurity()))
          continue;
        if (useQC) {
          if ((part->dz() * part->dz()) / (part->dzError() * part->dzError()) > 25.)
            continue;
          if ((part->dxy() * part->dxy()) / (part->dxyError() * part->dxyError()) < 25.)
            multWeighted += constWeight;
        } else
          multWeighted += constWeight;
      } else {
        if ((constWeight * part->pt()) < 1.0)
          continue;
        multWeighted += constWeight;
      }
    } else {
      auto part = static_cast<const reco::PFCandidate*>(cand);

      reco::TrackRef itrk = part->trackRef();
      if (itrk.isNonnull()) {  //Track exists --> charged particle
        auto vtxLead = vC->begin();
        auto vtxClose = vC->begin();  //Search for closest vertex to track
        for (auto vtx = vC->begin(); vtx != vC->end(); ++vtx) {
          if (fabs(itrk->dz(vtx->position())) < fabs(itrk->dz(vtxClose->position())))
            vtxClose = vtx;
        }
        if (!(vtxClose == vtxLead && itrk->quality(reco::TrackBase::qualityByName("highPurity"))))
          continue;

        if (useQC) {  //If useQC, require dz and d0 cuts
          float dz = itrk->dz(vtxClose->position());
          float d0 = itrk->dxy(vtxClose->position());
          float dz_sigma_square = pow(itrk->dzError(), 2) + pow(vtxClose->zError(), 2);
          float d0_sigma_square = pow(itrk->d0Error(), 2) + pow(vtxClose->xError(), 2) + pow(vtxClose->yError(), 2);
          if (dz * dz / dz_sigma_square > 25.)
            continue;
          if (d0 * d0 / d0_sigma_square < 25.)
            multWeighted += constWeight;
        } else
          multWeighted += constWeight;
      } else {  //No track --> neutral particle
        if ((constWeight * part->pt()) < 1.0)
          continue;  //Only use neutrals with pt > 1 GeV
        multWeighted += constWeight;
      }
    }

    float deta = daughter->eta() - jet->eta();
    float dphi = reco::deltaPhi(daughter->phi(), jet->phi());
    float partPt = constWeight * daughter->pt();
    float weight = partPt * partPt;

    sum_weight += weight;
    sum_pt += partPt;
    sum_deta += deta * weight;
    sum_dphi += dphi * weight;
    sum_deta2 += deta * deta * weight;
    sum_detadphi += deta * dphi * weight;
    sum_dphi2 += dphi * dphi * weight;
  }

  //Calculate axis2 and ptD
  float a = 0., b = 0., c = 0.;
  float ave_deta = 0., ave_dphi = 0., ave_deta2 = 0., ave_dphi2 = 0.;
  if (sum_weight > 0) {
    ave_deta = sum_deta / sum_weight;
    ave_dphi = sum_dphi / sum_weight;
    ave_deta2 = sum_deta2 / sum_weight;
    ave_dphi2 = sum_dphi2 / sum_weight;
    a = ave_deta2 - ave_deta * ave_deta;
    b = ave_dphi2 - ave_dphi * ave_dphi;
    c = -(sum_detadphi / sum_weight - ave_deta * ave_dphi);
  }

  int mult = std::round(multWeighted);
  float delta = sqrt(fabs((a - b) * (a - b) + 4 * c * c));
  float axis2 = (a + b - delta > 0 ? sqrt(0.5 * (a + b - delta)) : 0);
  float ptD = (sum_weight > 0 ? sqrt(sum_weight) / sum_pt : 0);
  return std::make_tuple(mult, ptD, axis2);
}

/// Descriptions method
void QGTagger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcJets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("srcRho", edm::InputTag("fixedGridRhoFastjetAll"));
  desc.add<bool>("computeLikelihood", true);
  desc.add<std::string>("jetsLabel", "QGL_AK4PFchs");
  desc.add<std::string>("systematicsLabel", "");
  desc.add<bool>("useQualityCuts", false);
  desc.add<edm::InputTag>("jec", edm::InputTag())->setComment("Jet correction service: only applied when non-empty");
  desc.add<edm::InputTag>("srcVertexCollection", edm::InputTag("offlinePrimaryVerticesWithBS"))
      ->setComment("Ignored for miniAOD, possible to keep empty");
  desc.add<edm::InputTag>("srcConstituentWeights", edm::InputTag())->setComment("Constituent weights ValueMap");
  descriptions.add("QGTagger", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(QGTagger);
