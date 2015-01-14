#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
#include "CondFormats/DataRecord/interface/QGLikelihoodRcd.h"
#include "CondFormats/DataRecord/interface/QGLikelihoodSystematicsRcd.h"

/**
 * EDProducer class to produced the qgLikelihood values and related variables
 * If the input jets are uncorrected, the jecService should be provided, so jet are corrected on the fly before the algorithm is applied
 * Authors: andrea.carlo.marini@cern.ch, tom.cornelis@cern.ch, cms-qg-workinggroup@cern.ch
 */
QGTagger::QGTagger(const edm::ParameterSet& iConfig) :
  jetsToken( 		consumes<edm::View<reco::Jet>>(		iConfig.getParameter<edm::InputTag>("srcJets"))),
  jetCorrectorToken(	consumes<reco::JetCorrector>(		iConfig.getParameter<edm::InputTag>("jec"))),
  vertexToken(		consumes<reco::VertexCollection>(	iConfig.getParameter<edm::InputTag>("srcVertexCollection"))),
  rhoToken(		consumes<double>(			iConfig.getParameter<edm::InputTag>("srcRho"))),
  jetsLabel(							iConfig.getParameter<std::string>("jetsLabel")),
  systLabel(							iConfig.getParameter<std::string>("systematicsLabel")),
  useQC(							iConfig.getParameter<bool>("useQualityCuts")),
  useJetCorr(							!iConfig.getParameter<edm::InputTag>("jec").label().empty()),
  produceSyst(							systLabel != "")
{
  produces<edm::ValueMap<float>>("qgLikelihood");
  produces<edm::ValueMap<float>>("axis2");
  produces<edm::ValueMap<int>>("mult");
  produces<edm::ValueMap<float>>("ptD");
  if(produceSyst){
    produces<edm::ValueMap<float>>("qgLikelihoodSmearedQuark");
    produces<edm::ValueMap<float>>("qgLikelihoodSmearedGluon");
    produces<edm::ValueMap<float>>("qgLikelihoodSmearedAll");
  }
  qgLikelihood = new QGLikelihoodCalculator();
  weStillNeedToCheckJetCandidates = true;
}


/// Produce qgLikelihood using {mult, ptD, -log(axis2)}
void QGTagger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  std::vector<float>* qgProduct 		= new std::vector<float>;
  std::vector<float>* axis2Product 		= new std::vector<float>;
  std::vector<int>*   multProduct 		= new std::vector<int>;
  std::vector<float>* ptDProduct 		= new std::vector<float>;
  std::vector<float>* smearedQuarkProduct 	= new std::vector<float>;
  std::vector<float>* smearedGluonProduct 	= new std::vector<float>;
  std::vector<float>* smearedAllProduct 	= new std::vector<float>;

  edm::Handle<edm::View<reco::Jet>> jets;				iEvent.getByToken(jetsToken, 		jets);
  edm::Handle<reco::JetCorrector> jetCorr;		if(useJetCorr)	iEvent.getByToken(jetCorrectorToken, 	jetCorr);
  edm::Handle<reco::VertexCollection> vertexCollection;			iEvent.getByToken(vertexToken, 		vertexCollection);
  edm::Handle<double> rho;						iEvent.getByToken(rhoToken, 		rho);

  edm::ESHandle<QGLikelihoodObject> QGLParamsColl;
  QGLikelihoodRcd const & rcdhandle = iSetup.get<QGLikelihoodRcd>();
  rcdhandle.get(jetsLabel, QGLParamsColl);

  edm::ESHandle<QGLikelihoodSystematicsObject> QGLSystColl;
  if(produceSyst){
    QGLikelihoodSystematicsRcd const & systrcdhandle = iSetup.get<QGLikelihoodSystematicsRcd>();
    systrcdhandle.get(systLabel, QGLSystColl);
  }

  for(auto jet = jets->begin(); jet != jets->end(); ++jet){
    float pt = (useJetCorr ? jet->pt()*jetCorr->correction(*jet) : jet->pt());

    float ptD, axis2; int mult;
    std::tie(mult, ptD, axis2) = calcVariables(&*jet, vertexCollection);

    float qgValue;
    if(mult > 2) qgValue = qgLikelihood->computeQGLikelihood(QGLParamsColl, pt, jet->eta(), *rho, {(float) mult, ptD, -std::log(axis2)});
    else         qgValue = -1;

    qgProduct->push_back(qgValue);
    if(produceSyst){
      smearedQuarkProduct->push_back(qgLikelihood->systematicSmearing(QGLSystColl, pt, jet->eta(), *rho, qgValue, 0));
      smearedGluonProduct->push_back(qgLikelihood->systematicSmearing(QGLSystColl, pt, jet->eta(), *rho, qgValue, 1));
      smearedAllProduct->push_back(qgLikelihood->systematicSmearing(  QGLSystColl, pt, jet->eta(), *rho, qgValue, 2));
    }
    axis2Product->push_back(axis2);
    multProduct->push_back(mult);
    ptDProduct->push_back(ptD);
  }

  putInEvent("qgLikelihood", jets, qgProduct,    iEvent);
  putInEvent("axis2",        jets, axis2Product, iEvent);
  putInEvent("mult",         jets, multProduct,  iEvent);
  putInEvent("ptD",          jets, ptDProduct,   iEvent);
  if(produceSyst){
    putInEvent("qgLikelihoodSmearedQuark", jets, smearedQuarkProduct, iEvent);
    putInEvent("qgLikelihoodSmearedGluon", jets, smearedGluonProduct, iEvent);
    putInEvent("qgLikelihoodSmearedAll",   jets, smearedAllProduct,   iEvent);
  }
}

/// Function to put product into event
template <typename T> void QGTagger::putInEvent(std::string name, const edm::Handle<edm::View<reco::Jet>>& jets, std::vector<T>* product, edm::Event& iEvent){
  std::auto_ptr<edm::ValueMap<T>> out(new edm::ValueMap<T>());
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(jets, product->begin(), product->end());
  filler.fill();
  iEvent.put(out, name);
  delete product;
}


/// Function to tell us if we are using packedCandidates, only test for first candidate
bool QGTagger::isPackedCandidate(const reco::Candidate* candidate){
  if(weStillNeedToCheckJetCandidates){
    if(typeid(pat::PackedCandidate)==typeid(*candidate)) weAreUsingPackedCandidates = true;
    else if(typeid(reco::PFCandidate)==typeid(*candidate)) weAreUsingPackedCandidates = false;
    else throw cms::Exception("WrongJetCollection", "Jet constituents are not particle flow candidates");
    weStillNeedToCheckJetCandidates = false;
  }
  return weAreUsingPackedCandidates;
}


/// Calculation of axis2, mult and ptD
std::tuple<int, float, float> QGTagger::calcVariables(const reco::Jet *jet, edm::Handle<reco::VertexCollection>& vC){
  float sum_weight = 0., sum_deta = 0., sum_dphi = 0., sum_deta2 = 0., sum_dphi2 = 0., sum_detadphi = 0., sum_pt = 0.;
  int mult = 0;

  //Loop over the jet constituents
  for(auto daughter : jet->getJetConstituentsQuick()){
    if(isPackedCandidate(daughter)){											//packed candidate situation
      auto part = static_cast<const pat::PackedCandidate*>(daughter);

      if(part->charge()){
        if(!(part->fromPV() > 1 && part->trackHighPurity())) continue;
        if(useQC){
          if((part->dz()*part->dz())/(part->dzError()*part->dzError()) > 25.) continue;
          if((part->dxy()*part->dxy())/(part->dxyError()*part->dxyError()) < 25.) ++mult;
        } else ++mult;
      } else {
        if(part->pt() < 1.0) continue;
        ++mult;
      }
    } else {
      auto part = static_cast<const reco::PFCandidate*>(daughter);

      reco::TrackRef itrk = part->trackRef();
      if(itrk.isNonnull()){												//Track exists --> charged particle
        auto vtxLead  = vC->begin();
        auto vtxClose = vC->begin();											//Search for closest vertex to track
        for(auto vtx = vC->begin(); vtx != vC->end(); ++vtx){
          if(fabs(itrk->dz(vtx->position())) < fabs(itrk->dz(vtxClose->position()))) vtxClose = vtx;
        }
        if(!(vtxClose == vtxLead && itrk->quality(reco::TrackBase::qualityByName("highPurity")))) continue;

        if(useQC){													//If useQC, require dz and d0 cuts
          float dz = itrk->dz(vtxClose->position());
          float d0 = itrk->dxy(vtxClose->position());
          float dz_sigma_square = pow(itrk->dzError(),2) + pow(vtxClose->zError(),2);
          float d0_sigma_square = pow(itrk->d0Error(),2) + pow(vtxClose->xError(),2) + pow(vtxClose->yError(),2);
          if(dz*dz/dz_sigma_square > 25.) continue;
          if(d0*d0/d0_sigma_square < 25.) ++mult;
        } else ++mult;
      } else {														//No track --> neutral particle
        if(part->pt() < 1.0) continue;											//Only use neutrals with pt > 1 GeV
        ++mult;
      }
    }

    float deta = daughter->eta() - jet->eta();
    float dphi = reco::deltaPhi(daughter->phi(), jet->phi());
    float partPt = daughter->pt();
    float weight = partPt*partPt;

    sum_weight += weight;
    sum_pt += partPt;
    sum_deta += deta*weight;
    sum_dphi += dphi*weight;
    sum_deta2 += deta*deta*weight;
    sum_detadphi += deta*dphi*weight;
    sum_dphi2 += dphi*dphi*weight;
  }

  //Calculate axis2 and ptD
  float a = 0., b = 0., c = 0.;
  float ave_deta = 0., ave_dphi = 0., ave_deta2 = 0., ave_dphi2 = 0.;
  if(sum_weight > 0){
    ave_deta = sum_deta/sum_weight;
    ave_dphi = sum_dphi/sum_weight;
    ave_deta2 = sum_deta2/sum_weight;
    ave_dphi2 = sum_dphi2/sum_weight;
    a = ave_deta2 - ave_deta*ave_deta;                          
    b = ave_dphi2 - ave_dphi*ave_dphi;                          
    c = -(sum_detadphi/sum_weight - ave_deta*ave_dphi);                
  }
  float delta = sqrt(fabs((a-b)*(a-b)+4*c*c));
  float axis2 = (a+b-delta > 0 ?  sqrt(0.5*(a+b-delta)) : 0);
  float ptD   = (sum_weight > 0 ? sqrt(sum_weight)/sum_pt : 0);
  return std::make_tuple(mult, ptD, axis2);
}


/// Descriptions method
void QGTagger::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcJets");
  desc.add<edm::InputTag>("srcRho");
  desc.add<std::string>("jetsLabel");
  desc.add<std::string>("systematicsLabel", "");
  desc.add<bool>("useQualityCuts");
  desc.add<edm::InputTag>("jec", edm::InputTag())->setComment("Jet correction service: only applied when non-empty");
  desc.add<edm::InputTag>("srcVertexCollection")->setComment("Ignored for miniAOD, possible to keep empty");
  descriptions.add("QGTagger",  desc);
}


//define this as a plug-in
DEFINE_FWK_MODULE(QGTagger);
