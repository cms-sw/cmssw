#ifndef PhysicsTools_PFCandProducer_FastJetAlgo
#define PhysicsTools_PFCandProducer_FastJetAlgo

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"

#include <iostream>

namespace pf2pat {
  

  //COLIN change name to FastJetAlgo

  class FastJetAlgo {

  public:
    typedef std::vector< fastjet::PseudoJet > PseudoJetCollection;
    typedef PseudoJetCollection::const_iterator PJI;
    
    // template it?
    typedef reco::PFCandidate           InputType;
    typedef std::vector<InputType>      InputCollection; 
    typedef edm::Handle< InputCollection > InputHandle;

    // template it?
    typedef reco::PFJet           JetType;
    typedef std::vector<JetType>  JetCollection; 
    typedef JetCollection::const_iterator JI;

    FastJetAlgo( const edm::ParameterSet& ps ); 
    
    /// get jet definition from parameter set
    void setJetDefinition( const edm::ParameterSet& ps);  

    /// set user defined jet definition
    void setJetDefinition( const fastjet::JetDefinition& jetDef);
    
    /// run the jet clustering on the input collection, and produce the reco jets
    const JetCollection& produce( const InputHandle& inputColl); 
    
    /// print internal pseudojets
    void printPseudoJets( std::ostream& out = std::cout) const;

    /// print output jets
    void printJets( std::ostream& out = std::cout) const;

    
  private:
    /// convert input elements from CMSSW (e.g. PFCandidates) 
    /// into fastjet input. 
    /// InputType could be a template parameter of the class
    void recoToFastJet(const InputCollection& inputColl); 

    /// run fast jet
    void runJetClustering(); 

    /// convert fastjet output to RECO data format (e.g. PFJet)
    const JetCollection& fastJetToReco();

    /// build a JetType (e.g. PFJet) from a pseudo-jet. Jet type
    /// could be a template parameter of the class. 
    JetType makeJet( const fastjet::PseudoJet& pseudoJet) const;

    /// build the vector< Ptr<Candidate> > pointing to constituents
    /// from the PseudoJet and the Handle information.
    reco::Jet::Constituents makeConstituents(const fastjet::PseudoJet& pseudoJet) const; 

    /// keep track of the input handle - set in the produce function.
    InputHandle          inputHandle_;

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
