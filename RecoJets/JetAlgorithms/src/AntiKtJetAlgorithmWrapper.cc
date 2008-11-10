// File: AntiKtJetAlgorithmWrapper.cc
// Description:  see FastJetProducer.h
// Author:  Andreas Oehler, University Karlsruhe (TH)
// Author:  Dorian Kcira, Institut de Physique Nucleaire,
//          Departement de Physique, Universite Catholique de Louvain
// Creation Date:  Nov. 06 2006 Initial version.
// Redesigned by F.Ratnikov  (UMd) Aug. 1, 2007
//--------------------------------------------

#include <string>

#include "fastjet/JetDefinition.hh"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoJets/JetAlgorithms/interface/AntiKtJetAlgorithmWrapper.h"

AntiKtJetAlgorithmWrapper::AntiKtJetAlgorithmWrapper(const edm::ParameterSet& fConfig)
  : FastJetBaseWrapper (fConfig)
{
  //configuring algorithm 
  double rParam = fConfig.getParameter<double> ("FJ_ktRParam");
  //choosing search-strategy:
  fastjet::Strategy fjStrategy = fastjet::plugin_strategy;
  std::string strategy = fConfig.getParameter<std::string>("Strategy");
  if (strategy == "Best") fjStrategy = fastjet::Best; // Chooses best Strategy for every event, depending on N and ktRParam
  else if (strategy == "N2Plain") fjStrategy = fastjet::N2Plain;  // N2Plain is best for N<50
  else if (strategy == "N2Tiled") fjStrategy = fastjet::N2Tiled;  // N2Tiled is best for 50<N<400
  else if (strategy == "N2MinHeapTiled") fjStrategy = fastjet::N2MinHeapTiled;  // N2MinHeapTiles is best for 400<N<15000
  else if (strategy == "NlnN") fjStrategy = fastjet::NlnN;  // NlnN is best for N>15000
  else if (strategy == "NlnNCam") fjStrategy = fastjet::NlnNCam;  // NlnNCam is best for N>6000
  else  edm::LogError("FastJetDefinition") << "AntiKtJetAlgorithmWrapper-> Unknown strategy: " 
					   << strategy
					   << ". Known strategies: Best N2Plain N2Tiled N2MinHeapTiled NlnN" << std::endl;
  //additional strategies are possible, but not documented in the manual as they are experimental,
  //they are also not used by the "Best" method. Note: "NlnNCam" only works with 
  //the cambridge_algorithm and does not link against CGAL, "NlnN" is only 
  //available if fastjet is linked against CGAL.
  //The above given numbers of N are for ktRaram=1.0, for other ktRParam the numbers would be 
  //different.
  
  mJetDefinition = new fastjet::JetDefinition (fastjet::antikt_algorithm, rParam, fjStrategy);
}

AntiKtJetAlgorithmWrapper::~AntiKtJetAlgorithmWrapper () {}

