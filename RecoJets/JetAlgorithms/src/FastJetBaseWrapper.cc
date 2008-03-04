#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceActiveArea.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoJets/JetAlgorithms/interface/FastJetBaseWrapper.h"



//  Base class wrapper around fastjet-package written by Matteo Cacciari and Gavin Salam.
//
//  The algorithms that underlie FastJet have required considerable
//  development and are described in hep-ph/0512210. If you use
//  FastJet as part of work towards a scientific publication, please
//  include a citation to the FastJet paper.
//
//  Also see: http://www.lpthe.jussieu.fr/~salam/fastjet/


FastJetBaseWrapper::FastJetBaseWrapper(const edm::ParameterSet& fConfig) 
  : mJetDefinition (0), 
    mActiveArea (0)
{
  mJetPtMin = fConfig.getParameter<double> ("JetPtMin");
  if (fConfig.getParameter<std::string> ("UE_Subtraction") == "yes") {            // accept pilup subtraction parameters
    double ghostEtaMax = fConfig.getParameter<double> ("Ghost_EtaMax");   //default Ghost_EtaMax should be 6
    int activeAreaRepeats = fConfig.getParameter<int> ("Active_Area_Repeats");   //default Active_Area_Repeats 5
    double ghostArea = fConfig.getParameter<double> ("GhostArea");   //default GhostArea 0.01
    mActiveArea = new fastjet::ActiveAreaSpec (ghostEtaMax, activeAreaRepeats, ghostArea);
  }
}


FastJetBaseWrapper::~FastJetBaseWrapper(){
  delete mJetDefinition;
  delete mActiveArea;
}

void FastJetBaseWrapper::run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput) {
  if (!fOutput) return;
  fOutput->clear();
  
  // convert inputs
  std::vector<fastjet::PseudoJet> fjInputs;
  
  JetReco::InputCollection::const_iterator input = fInput.begin();
  unsigned index = 0;
  for (; input != fInput.end(); ++input, ++index) {
    fjInputs.push_back (fastjet::PseudoJet ((*input)->px(),(*input)->py(),(*input)->pz(),(*input)->energy()));
    fjInputs.back().set_user_index(index);
  }
   
   // create an object that represents your choice of jet finder and 
   // the associated parameters
   // run the jet clustering with the above jet definition

   // here we need to keep both pointers, as "area" interfaces are missing in base class
   fastjet::ClusterSequenceActiveArea* clusterSequenceWithArea = 0;
   fastjet::ClusterSequenceWithArea* clusterSequence = 0;
   if (mActiveArea) {
     clusterSequenceWithArea = new fastjet::ClusterSequenceActiveArea (fjInputs, *mJetDefinition, *mActiveArea);
     clusterSequence = clusterSequenceWithArea;
   }
   else {
     clusterSequence = new fastjet::ClusterSequenceWithArea (fjInputs, *mJetDefinition);
   }
   // retrieve jets for selected mode
   std::vector<fastjet::PseudoJet> jets = clusterSequence->inclusive_jets (mJetPtMin);

   // get PU pt
   double median_Pt_Per_Area = clusterSequenceWithArea ? clusterSequenceWithArea->pt_per_unit_area() : 0.;

   // process found jets
   JetReco::InputCollection jetConstituents; // cache
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
     // reserve is not defined     jetConstituents.reserve (fastjet_constituents.size());
     jetConstituents.clear();
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


