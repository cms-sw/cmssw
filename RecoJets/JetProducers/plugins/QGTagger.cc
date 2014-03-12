/* Authors: andrea.carlo.marini@cern.ch, tom.cornelis@cern.ch
 */
#include <memory>
#include <TROOT.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "RecoJets/JetProducers/interface/QGTagger.h"
#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"


QGTagger::QGTagger(const edm::ParameterSet& iConfig) :
  srcJets        ( iConfig.getParameter<edm::InputTag>("srcJets")),
  srcRhoIso      ( iConfig.getParameter<edm::InputTag>("srcRhoIso")),
  jecService     ( iConfig.getParameter<std::string>("jec")),
  dataDir        ( TString(iConfig.getParameter<std::string>("dataDir"))),
  useCHS         ( iConfig.getParameter<bool>("useCHS"))
{
  produces<edm::ValueMap<float>>("qgLikelihood");
  produces<edm::ValueMap<float>>("axis2Likelihood");
  produces<edm::ValueMap<int>>("multLikelihood");
  produces<edm::ValueMap<float>>("ptDLikelihood");
  qgLikelihood 	= new QGLikelihoodCalculator(dataDir, useCHS);

  src_token=consumes<reco::PFJetCollection>(srcJets);
  rho_token=consumes<double>(srcRhoIso);
  vertex_token=consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVerticesWithBS"));
}


void QGTagger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  std::vector<float>* qgProduct = new std::vector<float>;
  std::vector<float>* axis2Product = new std::vector<float>;
  std::vector<int>* multProduct = new std::vector<int>;
  std::vector<float>* ptDProduct = new std::vector<float>;

  if(jecService != "") JEC = JetCorrector::getJetCorrector(jecService,iSetup);

  //Get rhokt6PFJets and primary vertex
  edm::Handle<double> rhoIso;
  iEvent.getByToken(rho_token, rhoIso);

  edm::Handle<reco::VertexCollection> vertexCollection;
  iEvent.getByToken(vertex_token, vertexCollection);

  edm::Handle<reco::PFJetCollection> pfJets;
  iEvent.getByToken(src_token, pfJets);

  for(reco::PFJetCollection::const_iterator pfJet = pfJets->begin(); pfJet != pfJets->end(); ++pfJet){
    if(jecService == "") pt = pfJet->pt();
    else pt = pfJet->pt()*JEC->correction(*pfJet, iEvent, iSetup);
    calcVariables(&*pfJet, vertexCollection);
    qgProduct->push_back(qgLikelihood->computeQGLikelihood2012(pt, pfJet->eta(), *rhoIso, mult, ptD, axis2));
    axis2Product->push_back(axis2);
    multProduct->push_back(mult);
    ptDProduct->push_back(ptD);
  }

  putInEvent("qgLikelihood", pfJets, qgProduct, iEvent);
  putInEvent("axis2Likelihood", pfJets, axis2Product, iEvent);
  putInEvent("multLikelihood", pfJets, multProduct, iEvent);
  putInEvent("ptDLikelihood", pfJets, ptDProduct, iEvent);
}

void QGTagger::putInEvent(std::string name, edm::Handle<reco::PFJetCollection> pfJets, std::vector<float>* product, edm::Event& iEvent){
  std::auto_ptr<edm::ValueMap<float>> out(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler(*out);
  filler.insert(pfJets, product->begin(), product->end());
  filler.fill();
  iEvent.put(out, name);
  delete product;
}

void QGTagger::putInEvent(std::string name, edm::Handle<reco::PFJetCollection> pfJets, std::vector<int>* product, edm::Event& iEvent){
  std::auto_ptr<edm::ValueMap<int>> out(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler filler(*out);
  filler.insert(pfJets, product->begin(), product->end());
  filler.fill();
  iEvent.put(out, name);
  delete product;
}


template <class jetClass> void QGTagger::calcVariables(const jetClass *jet, edm::Handle<reco::VertexCollection> vC){
  reco::VertexCollection::const_iterator vtxLead = vC->begin();

  float sum_weight = 0., sum_deta = 0., sum_dphi = 0., sum_deta2 = 0., sum_dphi2 = 0., sum_detadphi = 0., sum_pt = 0.;
  int nChg_QC = 0, nChg_ptCut = 0, nNeutral_ptCut = 0;

  //Loop over the jet constituents
  std::vector<reco::PFCandidatePtr> constituents = jet->getPFConstituents();
  for(unsigned i = 0; i < constituents.size(); ++i){
    reco::PFCandidatePtr part = jet->getPFConstituent(i);      
    if(!part.isNonnull()) continue;

    reco::TrackRef itrk = part->trackRef();;

    bool trkForAxis = false;
    if(itrk.isNonnull()){						//Track exists --> charged particle
      if(part->pt() > 1.0) nChg_ptCut++;

      //Search for closest vertex to track
      reco::VertexCollection::const_iterator vtxClose = vC->begin();
      for(reco::VertexCollection::const_iterator vtx = vC->begin(); vtx != vC->end(); ++vtx){
        if(fabs(itrk->dz(vtx->position())) < fabs(itrk->dz(vtxClose->position()))) vtxClose = vtx;
      }

      if(vtxClose == vtxLead){
        float dz = itrk->dz(vtxClose->position());
        float dz_sigma = sqrt(pow(itrk->dzError(),2) + pow(vtxClose->zError(),2));
	      
        if(itrk->quality(reco::TrackBase::qualityByName("highPurity")) && fabs(dz/dz_sigma) < 5.){
          trkForAxis = true;
          float d0 = itrk->dxy(vtxClose->position());
          float d0_sigma = sqrt(pow(itrk->d0Error(),2) + pow(vtxClose->xError(),2) + pow(vtxClose->yError(),2));
          if(fabs(d0/d0_sigma) < 5.) nChg_QC++;
        }
      }
    } else {								//No track --> neutral particle
      if(part->pt() > 1.0) nNeutral_ptCut++;
      trkForAxis = true;
    }
	  
    float deta = part->eta() - jet->eta();
    float dphi = 2*atan(tan(((part->phi()-jet->phi()))/2));           
    float partPt = part->pt(); 
    float weight = partPt*partPt;

    if(trkForAxis){							//Only use when trkForAxis
      sum_weight += weight;
      sum_pt += partPt;
      sum_deta += deta*weight;                  
      sum_dphi += dphi*weight;                                                                                             
      sum_deta2 += deta*deta*weight;                    
      sum_detadphi += deta*dphi*weight;                               
      sum_dphi2 += dphi*dphi*weight;
    }	
  }

  //Calculate axis and ptD
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


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void QGTagger::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcJets");
  desc.add<edm::InputTag>("srcRhoIso");
  desc.add<std::string>("dataDir");
  desc.add<std::string>("jec");
  desc.add<bool>("useCHS");
  descriptions.add("QGTagger", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(QGTagger);
