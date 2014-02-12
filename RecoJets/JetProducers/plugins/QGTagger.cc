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
#include "RecoJets/JetAlgorithms/interface/QGMLPCalculator.h"


QGTagger::QGTagger(const edm::ParameterSet& iConfig) :
  src            ( iConfig.getParameter<edm::InputTag>("srcJets")),
  srcRho         ( iConfig.getParameter<edm::InputTag>("srcRho")),
  srcRhoIso      ( iConfig.getParameter<edm::InputTag>("srcRhoIso")),
  jecService     ( iConfig.getUntrackedParameter<std::string>("jec","")),
  dataDir        ( TString(iConfig.getUntrackedParameter<std::string>("dataDir","RecoJets/JetProducers/data/"))), 
  useCHS         ( iConfig.getUntrackedParameter<Bool_t>("useCHS", false)),
  isPatJet	 ( iConfig.getUntrackedParameter<Bool_t>("isPatJet", false))
{
  for(TString product : {"qg","axis1", "axis2","mult","ptD"}){
    if(product != "axis1") produces<edm::ValueMap<Float_t>>((product + "Likelihood").Data());
    produces<edm::ValueMap<Float_t>>((product + "MLP").Data());
  }

  qgLikelihood 	= new QGLikelihoodCalculator(dataDir, useCHS);
  qgMLP		= new QGMLPCalculator("MLP", dataDir, true);
}


void QGTagger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  std::map<TString, std::vector<Float_t>* > products;
  for(TString product : {"qg","axis1", "axis2","mult","ptD"}){
    products[product + "Likelihood"] = new std::vector<Float_t>;
    products[product + "MLP"] = new std::vector<Float_t>;
  }

  if(jecService != "") JEC = JetCorrector::getJetCorrector(jecService,iSetup);

  //Get rhokt6PFJets and primary vertex
  edm::Handle<Double_t> rho, rhoIso;
  iEvent.getByLabel(srcRho, rho);
  variables["rho"] = (Float_t) *rho;
  iEvent.getByLabel(srcRhoIso, rhoIso);
  variables["rhoIso"] = (Float_t) *rhoIso;

  edm::Handle<reco::VertexCollection> vC_likelihood;
  iEvent.getByLabel("offlinePrimaryVerticesWithBS", vC_likelihood);
  edm::Handle<reco::VertexCollection> vC_MLP;
  iEvent.getByLabel("goodOfflinePrimaryVerticesQG", vC_MLP);

  edm::Handle<reco::PFJetCollection> pfJets;
  edm::Handle<std::vector<pat::Jet> > patJets;
  if(isPatJet) iEvent.getByLabel(src, patJets);
  else         iEvent.getByLabel(src, pfJets);

  if(isPatJet){
    for(std::vector<pat::Jet>::const_iterator patJet = patJets->begin(); patJet != patJets->end(); ++patJet){
      if(patJet->isPFJet()){
        variables["pt"] = patJet->pt();
        if((*vC_MLP.product()).size() > 0){
          calcVariables(&*patJet, vC_MLP, "MLP");
          products["qgMLP"]->push_back(qgMLP->QGvalue(variables));
        //  for(TString product : {"axis1", "axis2","mult","ptD"}) products[product + "MLP"]->push_back(variables[product]);
        } else products["qgMLP"]->push_back(-998);
	//in any case -- otherwise use if then ELSE in this case too
        for(TString product : {"axis1", "axis2","mult","ptD"}) products[product + "MLP"]->push_back(variables[product]);
        calcVariables(&*patJet, vC_likelihood, "Likelihood");
        products["qgLikelihood"]->push_back(qgLikelihood->QGvalue(variables));
        for(TString product : {"axis1", "axis2","mult","ptD"}) products[product + "Likelihood"]->push_back(variables[product]);
      } else {
        products["qgMLP"]->push_back(-997);
        products["qgLikelihood"]->push_back(-1);
      }
    } //loop on PAT jets
  } else { //NOT PAT
    for(reco::PFJetCollection::const_iterator pfJet = pfJets->begin(); pfJet != pfJets->end(); ++pfJet){
      if(jecService == "") variables["pt"] = pfJet->pt();
      else variables["pt"] = pfJet->pt()*JEC->correction(*pfJet, iEvent, iSetup);
      if((*vC_MLP.product()).size() > 0){
        calcVariables(&*pfJet, vC_MLP, "MLP");
        products["qgMLP"]->push_back(qgMLP->QGvalue(variables));
        //for(TString product : {"axis1", "axis2","mult","ptD"}) products[product + "MLP"]->push_back(variables[product]);
      } else products["qgMLP"]->push_back(-998);
	//in any case
        for(TString product : {"axis1", "axis2","mult","ptD"}) products[product + "MLP"]->push_back(variables[product]);
      calcVariables(&*pfJet, vC_likelihood, "Likelihood");
      products["qgLikelihood"]->push_back(qgLikelihood->QGvalue(variables));
      for(TString product : {"axis1", "axis2","mult","ptD"}) products[product + "Likelihood"]->push_back(variables[product]);
    }
  }


  for(std::map<TString, std::vector<Float_t>* >::iterator product = products.begin(); product != products.end(); ++product){
    if(product->first == "axis1Likelihood") continue;
    std::auto_ptr<edm::ValueMap<Float_t>> out(new edm::ValueMap<Float_t>());
    edm::ValueMap<Float_t>::Filler filler(*out);
    if(isPatJet) filler.insert(patJets, product->second->begin(), product->second->end());
    else 	 filler.insert(pfJets, product->second->begin(), product->second->end());
    filler.fill();
    iEvent.put(out, (product->first).Data());
    delete product->second;
  }
}


template <class jetClass> void QGTagger::calcVariables(const jetClass *jet, edm::Handle<reco::VertexCollection> vC, TString type){
  variables["eta"] = jet->eta();
  Bool_t useQC = true;
  if(fabs(jet->eta()) > 2.5 && type == "MLP") useQC = false;		//In MLP: no QC in forward region

  reco::VertexCollection::const_iterator vtxLead = vC->begin();

  Float_t sum_weight = 0., sum_deta = 0., sum_dphi = 0., sum_deta2 = 0., sum_dphi2 = 0., sum_detadphi = 0., sum_pt = 0.;
  Int_t nChg_QC = 0, nChg_ptCut = 0, nNeutral_ptCut = 0;

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
        Float_t dz = itrk->dz(vtxClose->position());
        Float_t dz_sigma = sqrt(pow(itrk->dzError(),2) + pow(vtxClose->zError(),2));
	      
        if(itrk->quality(reco::TrackBase::qualityByName("highPurity")) && fabs(dz/dz_sigma) < 5.){
          trkForAxis = true;
          Float_t d0 = itrk->dxy(vtxClose->position());
          Float_t d0_sigma = sqrt(pow(itrk->d0Error(),2) + pow(vtxClose->xError(),2) + pow(vtxClose->yError(),2));
          if(fabs(d0/d0_sigma) < 5.) nChg_QC++;
        }
      }
    } else {								//No track --> neutral particle
      if(part->pt() > 1.0) nNeutral_ptCut++;
      trkForAxis = true;
    }
	  
    Float_t deta = part->eta() - jet->eta();
    Float_t dphi = 2*atan(tan(((part->phi()-jet->phi()))/2));           
    Float_t partPt = part->pt(); 
    Float_t weight = partPt*partPt;

    if(!useQC || trkForAxis){					//If quality cuts, only use when trkForAxis
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
  Float_t a = 0., b = 0., c = 0.;
  Float_t ave_deta = 0., ave_dphi = 0., ave_deta2 = 0., ave_dphi2 = 0.;
  if(sum_weight > 0){
    variables["ptD"] = sqrt(sum_weight)/sum_pt;
    ave_deta = sum_deta/sum_weight;
    ave_dphi = sum_dphi/sum_weight;
    ave_deta2 = sum_deta2/sum_weight;
    ave_dphi2 = sum_dphi2/sum_weight;
    a = ave_deta2 - ave_deta*ave_deta;                          
    b = ave_dphi2 - ave_dphi*ave_dphi;                          
    c = -(sum_detadphi/sum_weight - ave_deta*ave_dphi);                
  } else variables["ptD"] = 0;
  Float_t delta = sqrt(fabs((a-b)*(a-b)+4*c*c));
  if(a+b+delta > 0) variables["axis1"] = sqrt(0.5*(a+b+delta));
  else variables["axis1"] = 0.;
  if(a+b-delta > 0) variables["axis2"] = sqrt(0.5*(a+b-delta));
  else variables["axis2"] = 0.;

  if(type == "MLP" && useQC) variables["mult"] = nChg_QC;
  else if(type == "MLP") variables["mult"] = (nChg_ptCut + nNeutral_ptCut);
  else variables["mult"] = (nChg_QC + nNeutral_ptCut);
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void QGTagger::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcJets");
  desc.add<edm::InputTag>("srcRho");
  desc.add<edm::InputTag>("srcRhoIso");
  desc.addUntracked<std::string>("dataDir","RecoJets/JetProducers/data/");
  desc.addUntracked<std::string>("jec","");
  desc.addUntracked<Bool_t>("useCHS", false);
  desc.addUntracked<Bool_t>("isPatJet", false);
  descriptions.add("QGTagger", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(QGTagger);
