// File: FastJetFWLiteWrapper.cc
// Description:  see FastJetProducer.h
// Author:  Andreas Oehler, University Karlsruhe (TH)
// Author:  Dorian Kcira, Institut de Physique Nucleaire,
//          Departement de Physique, Universite Catholique de Louvain
// Author:  Joanna Weng, IPP, ETH Zurich
// Creation Date:  Nov. 06 2006 Initial version.
//--------------------------------------------
 
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoParticleFlow/PFRootEvent/interface/ProtoJet.h"
#include "RecoParticleFlow/PFRootEvent/interface/FastJetFWLiteWrapper.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceActiveArea.hh"
#include <string.h>
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceActiveArea.hh"
#include "fastjet/ClusterSequenceArea.hh"
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

FastJetFWLiteWrapper::~FastJetFWLiteWrapper(){
  delete mJetDefinition;
  delete mActiveArea;
}

FastJetFWLiteWrapper::FastJetFWLiteWrapper()
  : mJetDefinition (0), 
    mActiveArea (0)
{

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
  mActiveArea = new fastjet::GhostedAreaSpec ( theGhost_EtaMax_, theActive_Area_Repeats_ , theGhostArea_);       

  
  //configuring algorithm 
  
  
//  fastjet::JetFinder jet_finder;
//  if (JetFinder=="cambridge_algorithm") jet_finder=fastjet::cambridge_algorithm;
//  else jet_finder=fastjet::kt_algorithm;
        
  //choosing search-strategy:
        
//  fastjet::Strategy strategy;
//  if (Strategy=="N2Plain") strategy=fastjet::N2Plain;
//  // N2Plain is best for N<50
//  else if (Strategy=="N2Tiled") strategy=fastjet::N2Tiled;
//  // N2Tiled is best for 50<N<400
//  else if (Strategy=="N2MinHeapTiled") strategy=fastjet::N2MinHeapTiled;
//  // N2MinHeapTiles is best for 400<N<15000
//  else if (Strategy=="NlnN") strategy=fastjet::NlnN;
//  // NlnN is best for N>15000
//  else if (Strategy=="NlnNCam") strategy=fastjet::NlnNCam;
//  // NlnNCam is best for N>6000
//  else strategy=fastjet::Best;
//  // Chooses best Strategy for every event, depending on N and ktRParam
        
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
        
  // theJetConfig_->theJetDef=fastjet::JetDefinition(jet_finder, theRparam_, strategy);
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

void FastJetFWLiteWrapper::run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput) {
  if (!fOutput) return;
  fOutput->clear();
  
  // convert inputs
  std::vector<fastjet::PseudoJet> fjInputs;
  fjInputs.reserve (fInput.size());
  
  for (unsigned i = 0; i < fInput.size(); ++i) {
    const JetReco::InputItem& c = fInput[i];
    fjInputs.push_back (fastjet::PseudoJet (c->px(),c->py(),c->pz(),c->energy()));
    fjInputs.back().set_user_index(i);
  }
   
  // create an object that represents your choice of jet finder and 
  // the associated parameters
  // run the jet clustering with the above jet definition

  // here we need to keep both pointers, as "area" interfaces are missing in base class
  fastjet::ClusterSequenceActiveArea* clusterSequenceWithArea = 0;
  fastjet::ClusterSequenceArea* clusterSequence = 0;
  //  if (mActiveArea) {
  // clusterSequenceWithArea = new fastjet::ClusterSequenceActiveArea (fjInputs, *mJetDefinition, *mActiveArea);
  // clusterSequence = clusterSequenceWithArea;
  // }
  // else {
  clusterSequence = new fastjet::ClusterSequenceArea (fjInputs, *mJetDefinition, *mActiveArea);
  // }
  // retrieve jets for selected mode
  std::vector<fastjet::PseudoJet> jets = clusterSequence->inclusive_jets (thePtMin_);

  // get PU pt
  double median_Pt_Per_Area = clusterSequenceWithArea ? clusterSequenceWithArea->pt_per_unit_area() : 0.;

  // process found jets
  for (std::vector<fastjet::PseudoJet>::const_iterator jet=jets.begin(); jet!=jets.end();++jet) {
    // jet itself
    double px=jet->px();
    double py=jet->py();
    double pz=jet->pz();
    double E=jet->E();
    double jetArea=clusterSequence->area(*jet);
    double pu=0.;
    // PU subtraction
    if (clusterSequenceWithArea) {
      fastjet::PseudoJet pu_p4 = median_Pt_Per_Area * clusterSequenceWithArea->area_4vector(*jet);
      pu = pu_p4.E();
      if (pu_p4.perp2() >= jet->perp2() || pu_p4.E() >= jet->E()) { // if the correction is too large, set the jet to zero
        px = py = pz = E = 0.;
      } 
      else {   // otherwise do an E-scheme subtraction
        px -= pu_p4.px();
        py -= pu_p4.py();
        pz -= pu_p4.pz();
        E -= pu_p4.E();
      }
    }
    math::XYZTLorentzVector p4(px,py,pz,E);
    // constituents
    std::vector<fastjet::PseudoJet> fastjet_constituents = clusterSequence->constituents(*jet);
    JetReco::InputCollection jetConstituents; 
    jetConstituents.reserve (fastjet_constituents.size());
    for (std::vector<fastjet::PseudoJet>::const_iterator itConst=fastjet_constituents.begin();
         itConst!=fastjet_constituents.end();itConst++){
      jetConstituents.push_back(fInput[(*itConst).user_index()]);
    }
    // Build ProtoJet
    fOutput->push_back(ProtoJet(p4,jetConstituents));
    fOutput->back().setJetArea (jetArea);
    fOutput->back().setPileup (pu);
  }
  // cleanup
  if (clusterSequenceWithArea) delete clusterSequenceWithArea;
  else delete clusterSequence; // sigh... No plymorphism in fastjet
}


