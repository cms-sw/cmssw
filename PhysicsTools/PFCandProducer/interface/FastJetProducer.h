#ifndef PhysicsTools_PFCandProducer_FastJetProducer
#define PhysicsTools_PFCandProducer_FastJetProducer

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"

#include <iostream>

namespace pf2pat {
  

  //COLIN change name to FastJetAlgo

  class FastJetProducer {

  public:
    typedef std::vector< fastjet::PseudoJet > PseudoJetCollection;
    typedef reco::PFCandidateCollection InputCollection; 
    typedef reco::PFJetCollection JetCollection; 

    FastJetProducer( const edm::ParameterSet& ps ); 
    
    /// get jet definition from parameter set
    void setJetDefinition( const edm::ParameterSet& ps);  

    /// get user defined jet definition
    void setJetDefinition( const fastjet::JetDefinition& jetDef) {
      jetDefinition_ = jetDef; 
    }
    
    /// run the jet clustering on the input collection, and produce the reco jets
    const JetCollection& produce( const InputCollection& inputColl); 
    
    /// print internal pseudojets
    void printPseudoJets( std::ostream& out = std::cout) const;

    
  private:
    /// convert input elements from CMSSW (e.g. PFCandidates) 
    /// into fastjet input. could be a function template. 
    void recoToFastJet(const InputCollection& inputColl); 

    /// run fast jet
    void runJetClustering(); 

    /// convert fastjet output to RECO data format (e.g. PFJet)
    const JetCollection& fastJetToReco();

    /// fastjet input
    PseudoJetCollection  input_;
    
    /// fastjet output
    PseudoJetCollection  output_;

    /// output jet collection
    JetCollection jetCollection_; 

    /// definition of the algorithm, and of the algorithm parameters
    fastjet::JetDefinition  jetDefinition_;

    /// cluster sequence
    fastjet::ClusterSequence* clusterSequence_;

  };

}

#endif
