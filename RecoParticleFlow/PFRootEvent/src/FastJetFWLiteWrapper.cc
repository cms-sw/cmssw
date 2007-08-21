// File: FastJetFWLiteWrapper.cc
// Description:  see FastJetProducer.h
// Author:  Andreas Oehler, University Karlsruhe (TH)
// Author:  Dorian Kcira, Institut de Physique Nucleaire,
//          Departement de Physique, Universite Catholique de Louvain
// Author:  Joanna Weng, IPP, ETH Zurich
// Creation Date:  Nov. 06 2006 Initial version.
//--------------------------------------------

#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "RecoParticleFlow/PFRootEvent/interface/FastJetFWLiteWrapper.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceActiveArea.hh"
#include <string.h>

//  Wrapper around fastjet-package written by Matteo Cacciari and Gavin Salam.
//
//  The algorithms that underlie FastJet have required considerable
//  development and are described in hep-ph/0512210. If you use
//  FastJet as part of work towards a scientific publication, please
//  include a citation to the FastJet paper.
//
//  Also see: http://www.lpthe.jussieu.fr/~salam/fastjet/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

struct FastJetFWLiteWrapper::JetConfig{
  fastjet::JetDefinition  theJetDef;
  fastjet::ActiveAreaSpec theAreaSpec;
};


FastJetFWLiteWrapper::~FastJetFWLiteWrapper(){
  delete theJetConfig_;
}

FastJetFWLiteWrapper::FastJetFWLiteWrapper(){
  //Getting information from the ParameterSet (reading config files):
  theJetConfig_=0;
  //default ktRParam should be 1
  theRparam_=1;
  //default PtMin should be 10
  thePtMin_=10.;
  //default JetFinder should be "kt_algorithm"
  string JetFinder="kt_algorithm";; 
  //default Strategy should be "Best" 
  string Strategy="Best";
  //default dcut should be -1 (off)
  theDcut_=-1.;
  //default njets should be -1 (off)
  theNjets_=-1;  
  //default UE_Subtraction should be false (off)
  theDoSubtraction_=false; 
  //default Ghost_EtaMax should be 6
  theGhost_EtaMax_=6;
  //default Active_Area_Repeats 5
  theActive_Area_Repeats_=5;
  //default GhostArea 0.01  
  theGhostArea_=0.01;
	
  theJetConfig_=new JetConfig;
  theJetConfig_->theAreaSpec=fastjet::ActiveAreaSpec(theGhost_EtaMax_, theActive_Area_Repeats_, theGhostArea_);
  
  //configuring algorithm 
  
  
  fastjet::JetFinder jet_finder;
  if (JetFinder=="cambridge_algorithm") jet_finder=fastjet::cambridge_algorithm;
  else jet_finder=fastjet::kt_algorithm;
	
  //choosing search-strategy:
	
  fastjet::Strategy strategy;
  if (Strategy=="N2Plain") strategy=fastjet::N2Plain;
  // N2Plain is best for N<50
  else if (Strategy=="N2Tiled") strategy=fastjet::N2Tiled;
  // N2Tiled is best for 50<N<400
  else if (Strategy=="N2MinHeapTiled") strategy=fastjet::N2MinHeapTiled;
  // N2MinHeapTiles is best for 400<N<15000
  else if (Strategy=="NlnN") strategy=fastjet::NlnN;
  // NlnN is best for N>15000
  else if (Strategy=="NlnNCam") strategy=fastjet::NlnNCam;
  // NlnNCam is best for N>6000
  else strategy=fastjet::Best;
  // Chooses best Strategy for every event, depending on N and ktRParam
	
  //additional strategies are possible, but not documented in the manual as they are experimental,
  //they are also not used by the "Best" method. Note: "NlnNCam" only works with 
  //the cambridge_algorithm and does not link against CGAL, "NlnN" is only 
  //available if fastjet is linked against CGAL.
  //The above given numbers of N are for ktRaram=1.0, for other ktRParam the numbers would be 
  //different.
  
  theMode_=0;
  if ((theNjets_!=-1)&&(theDcut_==-1)){
    theMode_=3;
  }
  else if ((theNjets_==-1)&&(theDcut_!=-1)){
    theMode_=2;
  }
  else if ((theNjets_!=-1)&&(theDcut_!=-1)){
    cout  <<"[FastJetWrapper] `njets` and `dcut` set!=-1! - running inclusive Mode"<<endl;
    theMode_=1;
  }
  else {
    theMode_=0;     
  }
	
  theJetConfig_->theJetDef=fastjet::JetDefinition(jet_finder, theRparam_, strategy);
  cout <<"*******************************************"<<endl;
  cout <<"* Configuration of FastJet                "<<endl;
  if (theDoSubtraction_) cout <<"* running with ActiveAreaSubtraction(median)"<<endl;
  switch (theMode_){
  case 0:
    cout <<"* Mode     : inclusive"<<endl;
    cout <<"* PtMin    : "<<thePtMin_<<endl;
    break;
  case 1:
    cout <<"* [WARNING] Mode     : inclusive - dcut and njets set!"<<endl;
    cout <<"* PtMin    : "<<thePtMin_<<endl;
    break;
  case 2:    
    cout <<"* Mode     : exclusive"<<endl;
    cout <<"* dcut     : "<<theDcut_<<endl;
    break;
  case 3:
    cout <<"* Mode     : exclusive"<<endl;
    cout <<"* njets    : "<<theNjets_<<endl;
    break;
  }
  cout <<"* Rparam   : "<<theRparam_<<endl;
  cout <<"* JetFinder: "<<JetFinder<<endl;
  cout <<"* Strategy : "<<Strategy<<endl;
  cout <<"*******************************************"<<endl;
  
}

//void FastJetFWLiteWrapper::run(const std::vector <const reco::Candidate*>& fInput, 
//			  std::vector<ProtoJet>* fOutput){
void FastJetFWLiteWrapper::run(const std::vector<FJCand>& fInput, 
			       std::vector<ProtoJet>* fOutput) {
		
  std::vector<fastjet::PseudoJet> input_vectors;
  int index_=0;
  for (std::vector<FJCand>::const_iterator inputCand=fInput.begin();
       inputCand!=fInput.end();inputCand++){
      
    double px=(*inputCand)->px();
    double py=(*inputCand)->py();
    double pz=(*inputCand)->pz();
    double E=(*inputCand)->energy();
    fastjet::PseudoJet PsJet(px,py,pz,E);
    PsJet.set_user_index(index_);
    input_vectors.push_back(PsJet);
    index_++;
  }
		
  // create an object that represents your choice of jet finder and 
  // the associated parameters
  // run the jet clustering with the above jet definition
		
  std::vector<fastjet::PseudoJet> theJets;
		
  if (theDoSubtraction_) {
    //with subtraction
    fastjet::ClusterSequenceActiveArea clust_AAseq(input_vectors,theJetConfig_->theJetDef,theJetConfig_->theAreaSpec);
			
    //  cout<< "***************************"<<endl;
    //  cout<< "* Strategy adopted by FastJet for this event was "<<
    //clust_seq.strategy_string()<<endl;
    //  cout<< "***************************\n"<<endl;
    //get jets
			
    //select mode:
			
    if (theMode_==0){
      theJets=clust_AAseq.inclusive_jets(thePtMin_);
    }
    else if (theMode_==3){
      theJets=clust_AAseq.exclusive_jets(theNjets_);
    }
    else if (theMode_==2){
      theJets=clust_AAseq.exclusive_jets(theDcut_);
    }
    else if ((theNjets_!=-1)&&(theDcut_!=-1)){
      cout <<"[FastJetWrapper] `njets` and `dcut` set!=-1! - running inclusive Mode"<<endl;
      theJets=clust_AAseq.inclusive_jets(thePtMin_);
    }
    else {
      //default mode
      theJets=clust_AAseq.inclusive_jets(thePtMin_);
    }
			
    //make CMSSW-Objects:
    std::vector<FJCand> jetConst;
    theMedian_Pt_Per_Area_=clust_AAseq.pt_per_unit_area();
    for (std::vector<fastjet::PseudoJet>::const_iterator itJet=theJets.begin();
	 itJet!=theJets.end();itJet++){
      std::vector<fastjet::PseudoJet> jet_constituents = clust_AAseq.constituents(*itJet);
      for (std::vector<fastjet::PseudoJet>::const_iterator itConst=jet_constituents.begin();
	   itConst!=jet_constituents.end();itConst++){
	jetConst.push_back(fInput[(*itConst).user_index()]);
      }
      fastjet::PseudoJet corrected_jet;
      fastjet::PseudoJet area_4vect = theMedian_Pt_Per_Area_ * clust_AAseq.area_4vector(*itJet);
      if (area_4vect.perp2() >= (*itJet).perp2() || 
	  area_4vect.E() >= (*itJet).E()) { 
	// if the correction is too large, set the jet to zero
	corrected_jet = 0.0 * (*itJet);
      } 
      else {   // otherwise do an E-scheme subtraction
	corrected_jet = (*itJet) - area_4vect;
      }
				
      double px=corrected_jet.px();
      double py=corrected_jet.py();
      double pz=corrected_jet.pz();
      double E=corrected_jet.E();
      math::XYZTLorentzVector p4(px,py,pz,E);
      fOutput->push_back(ProtoJet(p4,jetConst));
      jetConst.clear();
    }
  }
		
		
  // or run without subtraction:
  else {
    fastjet::ClusterSequence clust_seq(input_vectors, theJetConfig_->theJetDef);
			
    //select mode:
    if ((theNjets_==-1)&&(theDcut_==-1)){
      theJets=clust_seq.inclusive_jets(thePtMin_);
    }
    else if ((theNjets_!=-1)&&(theDcut_==-1)){
      theJets=clust_seq.exclusive_jets(theNjets_);
    }
    else if ((theNjets_==-1)&&(theDcut_!=-1)){
      theJets=clust_seq.exclusive_jets(theDcut_);
    }
    else if ((theNjets_!=-1)&&(theDcut_!=-1)){
      cout  <<"[FastJetWrapper] `njets` and `dcut` set!=-1! - running inclusive Mode"<<endl;
      theJets=clust_seq.inclusive_jets(thePtMin_);
    }
    else {
      //default mode
      theJets=clust_seq.inclusive_jets(thePtMin_);
    }
			
    //make CMSSW-Objects:
    std::vector<FJCand> jetConst;
    for (std::vector<fastjet::PseudoJet>::const_iterator itJet=theJets.begin();
	 itJet!=theJets.end();itJet++){
      std::vector<fastjet::PseudoJet> jet_constituents = clust_seq.constituents(*itJet);
      for (std::vector<fastjet::PseudoJet>::const_iterator itConst=jet_constituents.begin();
	   itConst!=jet_constituents.end();itConst++){
	jetConst.push_back(fInput[(*itConst).user_index()]);
					
      }
      double px=(*itJet).px();
      double py=(*itJet).py();
      double pz=(*itJet).pz();
      double E=(*itJet).E();
      math::XYZTLorentzVector p4(px,py,pz,E);
      fOutput->push_back(ProtoJet(p4,jetConst));
      jetConst.clear();
    }
  }
}
	
	
