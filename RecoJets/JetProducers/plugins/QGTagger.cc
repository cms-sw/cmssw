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
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
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
  srcJets        ( iConfig.getParameter<edm::InputTag>("srcJets")),
  srcRho         ( iConfig.getParameter<edm::InputTag>("srcRho")),
  srcVertexC     ( iConfig.getParameter<edm::InputTag>("srcVertexCollection")),
  jetsLabel	 ( iConfig.getParameter<std::string>("jetsLabel")),
  jecService     ( iConfig.getParameter<std::string>("jec")),
  systLabel	 ( iConfig.getParameter<std::string>("systematicsLabel"))
{
  useJEC = (jecService != "");
  produceSyst = (systLabel != "");

  produces<edm::ValueMap<float>>("qgLikelihood");
  produces<edm::ValueMap<float>>("axis2Likelihood");
  produces<edm::ValueMap<int>>("multLikelihood");
  produces<edm::ValueMap<float>>("ptDLikelihood");
  if(produceSyst){
    produces<edm::ValueMap<float>>("qgLikelihoodSmearedQuark");
    produces<edm::ValueMap<float>>("qgLikelihoodSmearedGluon");
    produces<edm::ValueMap<float>>("qgLikelihoodSmearedAll");
  }

  jets_token	= consumes<reco::PFJetCollection>(srcJets);
  rho_token	= consumes<double>(srcRho);
  vertex_token	= consumes<reco::VertexCollection>(srcVertexC);

  qgLikelihood 	= new QGLikelihoodCalculator();
}


/// Produce qgLikelihood using {mult, ptD, -log(axis2)}
void QGTagger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  std::vector<float>* qgProduct = new std::vector<float>;
  std::vector<float>* axis2Product = new std::vector<float>;
  std::vector<int>* multProduct = new std::vector<int>;
  std::vector<float>* ptDProduct = new std::vector<float>;
  std::vector<float>* smearedQuarkProduct = new std::vector<float>;
  std::vector<float>* smearedGluonProduct = new std::vector<float>;
  std::vector<float>* smearedAllProduct = new std::vector<float>;

  if(useJEC) JEC = JetCorrector::getJetCorrector(jecService, iSetup);

  edm::Handle<double> rho;
  iEvent.getByToken(rho_token, rho);

  edm::Handle<reco::VertexCollection> vertexCollection;
  iEvent.getByToken(vertex_token, vertexCollection);

  edm::Handle<reco::PFJetCollection> pfJets;
  iEvent.getByToken(jets_token, pfJets);

  edm::ESHandle<QGLikelihoodObject> QGLParamsColl;
  QGLikelihoodRcd const & rcdhandle = iSetup.get<QGLikelihoodRcd>();
  rcdhandle.get(jetsLabel, QGLParamsColl);

  edm::ESHandle<QGLikelihoodSystematicsObject> QGLSystColl;
  if(produceSyst){
    QGLikelihoodSystematicsRcd const & systrcdhandle = iSetup.get<QGLikelihoodSystematicsRcd>();
    systrcdhandle.get(systLabel, QGLSystColl);
  }

  for(auto pfJet = pfJets->begin(); pfJet != pfJets->end(); ++pfJet){
    if(useJEC) pt = pfJet->pt()*JEC->correction(*pfJet, iEvent, iSetup);
    else pt = pfJet->pt();
    calcVariables(&*pfJet, vertexCollection);
    float qgValue = qgLikelihood->computeQGLikelihood(QGLParamsColl, pt, pfJet->eta(), *rho, {(float)mult, ptD, -std::log(axis2)});
    qgProduct->push_back(qgValue);
    if(produceSyst){
      smearedQuarkProduct->push_back(qgLikelihood->systematicSmearing(QGLSystColl, pt, pfJet->eta(), *rho, qgValue, 0));
      smearedGluonProduct->push_back(qgLikelihood->systematicSmearing(QGLSystColl, pt, pfJet->eta(), *rho, qgValue, 1));
      smearedAllProduct->push_back(qgLikelihood->systematicSmearing(QGLSystColl, pt, pfJet->eta(), *rho, qgValue, 2));
    }
    axis2Product->push_back(axis2);
    multProduct->push_back(mult);
    ptDProduct->push_back(ptD);
  }

  putInEvent("qgLikelihood", pfJets, qgProduct, iEvent);
  putInEvent("axis2Likelihood", pfJets, axis2Product, iEvent);
  putInEvent("multLikelihood", pfJets, multProduct, iEvent);
  putInEvent("ptDLikelihood", pfJets, ptDProduct, iEvent);
  if(produceSyst){
    putInEvent("qgLikelihoodSmearedQuark", pfJets, smearedQuarkProduct, iEvent);
    putInEvent("qgLikelihoodSmearedGluon", pfJets, smearedGluonProduct, iEvent);
    putInEvent("qgLikelihoodSmearedAll", pfJets, smearedAllProduct, iEvent);
  }
}

/// Function to put product into event
template <typename T> void QGTagger::putInEvent(std::string name, edm::Handle<reco::PFJetCollection> pfJets, std::vector<T>* product, edm::Event& iEvent){
  std::auto_ptr<edm::ValueMap<T>> out(new edm::ValueMap<T>());
  typename edm::ValueMap<T>::Filler filler(*out);
  filler.insert(pfJets, product->begin(), product->end());
  filler.fill();
  iEvent.put(out, name);
  delete product;
}


/// Calculation of axis2, mult and ptD
template <class jetClass> void QGTagger::calcVariables(const jetClass *jet, edm::Handle<reco::VertexCollection> vC){
  reco::VertexCollection::const_iterator vtxLead = vC->begin();

  float sum_weight = 0., sum_deta = 0., sum_dphi = 0., sum_deta2 = 0., sum_dphi2 = 0., sum_detadphi = 0., sum_pt = 0.;
  int nChg_QC = 0, nNeutral_ptCut = 0;

  //Loop over the jet constituents
  for(auto part : jet->getPFConstituents()){
    if(!part.isNonnull()) continue;
    reco::TrackRef itrk = part->trackRef();;

    bool usePart = false;
    if(itrk.isNonnull()){												//Track exists --> charged particle
      reco::VertexCollection::const_iterator vtxClose = vC->begin();							//Search for closest vertex to track
      for(auto vtx = vC->begin(); vtx != vC->end(); ++vtx){
        if(fabs(itrk->dz(vtx->position())) < fabs(itrk->dz(vtxClose->position()))) vtxClose = vtx;
      }

      if(vtxClose == vtxLead){
        float dz = itrk->dz(vtxClose->position());
        float dz_sigma = sqrt(pow(itrk->dzError(),2) + pow(vtxClose->zError(),2));
	      
        if(itrk->quality(reco::TrackBase::qualityByName("highPurity")) && fabs(dz/dz_sigma) < 5.){
          usePart = true;												//Use charged particles if high purity track
          float d0 = itrk->dxy(vtxClose->position());
          float d0_sigma = sqrt(pow(itrk->d0Error(),2) + pow(vtxClose->xError(),2) + pow(vtxClose->yError(),2));
          if(fabs(d0/d0_sigma) < 5.) nChg_QC++;
        }
      }
    } else {														//No track --> neutral particle
      if(part->pt() > 1.0) nNeutral_ptCut++;
      usePart = true;													//Use all neutral particles
    }
	  
    if(usePart){
      float deta = part->eta() - jet->eta();
      float dphi = reco::deltaPhi(part->phi(), jet->phi());
      float partPt = part->pt(); 
      float weight = partPt*partPt;

      sum_weight += weight;
      sum_pt += partPt;
      sum_deta += deta*weight;                  
      sum_dphi += dphi*weight;                                                                                             
      sum_deta2 += deta*deta*weight;                    
      sum_detadphi += deta*dphi*weight;                               
      sum_dphi2 += dphi*dphi*weight;
    }	
  }

  //Calculate axis2 and ptD
  float a = 0., b = 0., c = 0.;
  float ave_deta = 0., ave_dphi = 0., ave_deta2 = 0., ave_dphi2 = 0.;
  if(sum_weight > 0){
    ptD = sqrt(sum_weight)/sum_pt;
    ave_deta = sum_deta/sum_weight;
    ave_dphi = sum_dphi/sum_weight;
    ave_deta2 = sum_deta2/sum_weight;
    ave_dphi2 = sum_dphi2/sum_weight;
    a = ave_deta2 - ave_deta*ave_deta;                          
    b = ave_dphi2 - ave_dphi*ave_dphi;                          
    c = -(sum_detadphi/sum_weight - ave_deta*ave_dphi);                
  } else ptD = 0;
  float delta = sqrt(fabs((a-b)*(a-b)+4*c*c));
  if(a+b-delta > 0) axis2 = sqrt(0.5*(a+b-delta));
  else axis2 = 0.;

  mult = (nChg_QC + nNeutral_ptCut);
}


/// Descriptions method
void QGTagger::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcJets");
  desc.add<edm::InputTag>("srcRho");
  desc.add<edm::InputTag>("srcVertexCollection");
  desc.add<std::string>("jetsLabel");
  desc.add<std::string>("jec");
  desc.add<std::string>("systematicsLabel");
  descriptions.add("QGTagger", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(QGTagger);
