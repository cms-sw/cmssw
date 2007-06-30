// File: FastJetAlgorithmWrapper.cc
// Description:  see FastJetProducer.h
// Author:  Andreas Oehler, University Karlsruhe (TH)
// Author:  Dorian Kcira, Institut de Physique Nucleaire,
//          Departement de Physique, Universite Catholique de Louvain
// Creation Date:  Nov. 06 2006 Initial version.
//--------------------------------------------


#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"

#include "RecoJets/JetAlgorithms/interface/SISConeAlgorithmWrapper.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/SISConePlugin.hh"
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

SISConeAlgorithmWrapper::SISConeAlgorithmWrapper () {
  
  mPlugin = new fastjet::SISConePlugin (0.5, 0.5, 0, 0., false, fastjet::SISConePlugin::SM_pttilde);
}

SISConeAlgorithmWrapper::SISConeAlgorithmWrapper (
						  double fConeRadius, 
						  double fConeOverlapThreshold,
						  const std::string& fSplitMergeScale,
						  int fMaxPasses, 
						  double fProtojetPtMin,
						  bool fCaching,
						  int fDebug
						  )
{
  fastjet::SISConePlugin::SplitMergeScale scale = fastjet::SISConePlugin::SM_pttilde;
  if (fSplitMergeScale == "pt") scale = fastjet::SISConePlugin::SM_pt;
  else if (fSplitMergeScale == "Et") scale = fastjet::SISConePlugin::SM_Et;
  else if (fSplitMergeScale == "mt") scale = fastjet::SISConePlugin::SM_mt;
  else if (fSplitMergeScale != "pttilde") {
    std::cerr << "SISConeAlgorithmWrapper::SISConeAlgorithmWrapper-> Unknown scale " << fSplitMergeScale
	      << ". Known scales are: pt Et mt pttilde" << std::cerr;
  }
    
  mPlugin = new fastjet::SISConePlugin (fConeRadius, 
					fConeOverlapThreshold, 
					fMaxPasses,
					fProtojetPtMin,
					fCaching,
					scale);
}


SISConeAlgorithmWrapper::~SISConeAlgorithmWrapper () {
  delete mPlugin;
}

void SISConeAlgorithmWrapper::run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput) {
  
  std::vector<fastjet::PseudoJet> input_vectors;
  int index_=0;
  for (JetReco::InputCollection::const_iterator inputCand=fInput.begin();
       inputCand!=fInput.end();++inputCand){ 
    
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

  // create a jet-definition based on the plugin
  fastjet::JetDefinition jetDefinition (mPlugin);
  
  // run the jet clustering with the above jet definition
  fastjet::ClusterSequence clusterSequence (input_vectors, jetDefinition);

  // extract all inclusive jets
  vector<fastjet::PseudoJet> inclusiveJets = clusterSequence.inclusive_jets (0);

  //make CMSSW-Objects:
  for (std::vector<fastjet::PseudoJet>::const_iterator itJet=inclusiveJets.begin();
       itJet!=inclusiveJets.end();++itJet){
    JetReco::InputCollection cmsJetConstituents;
    std::vector<fastjet::PseudoJet> fastjetConstituents = clusterSequence.constituents(*itJet);
    for (std::vector<fastjet::PseudoJet>::const_iterator itConst=fastjetConstituents.begin();
	 itConst!=fastjetConstituents.end();++itConst){
      cmsJetConstituents.push_back(fInput[itConst->user_index()]);
    }
    double px=(*itJet).px();
    double py=(*itJet).py();
    double pz=(*itJet).pz();
    double E=(*itJet).E();
    math::XYZTLorentzVector p4(px,py,pz,E);
    fOutput->push_back(ProtoJet(p4,cmsJetConstituents));
  }
}
