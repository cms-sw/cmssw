#include "CommonTools/ParticleFlow/interface/FastJetAlgo.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "RecoJets/JetProducers/interface/JetSpecific.h"

#include "fastjet/ClusterSequence.hh"


using namespace pf2pat;
using namespace fastjet;
using namespace std;


FastJetAlgo::FastJetAlgo( const edm::ParameterSet& ps ) 
  : clusterSequence_(0) {
  setJetDefinition( ps ); 
}

void FastJetAlgo::setJetDefinition( const edm::ParameterSet& ps) {
  // here extract parameter set info to make the jet definition

  unsigned algorithm = ps.getParameter<unsigned>("algorithm");
  double distance = ps.getParameter<double>("distance");

  JetDefinition jetDef( static_cast<JetAlgorithm>(algorithm), distance);
  setJetDefinition( jetDef ); 
}


void FastJetAlgo::setJetDefinition( const fastjet::JetDefinition& jetDef) {
  cout<<jetDef.description()<<endl;
  jetDefinition_ = jetDef; 
}


const FastJetAlgo::JetCollection& FastJetAlgo::produce( const FastJetAlgo::InputHandle& inputHandle) {
  
  // the input handle will be necessary to build the Ptrs to the jet constituents.
  inputHandle_ = inputHandle;
  const InputCollection& inputColl = *inputHandle; 
  recoToFastJet( inputColl );
  runJetClustering();
  return fastJetToReco(); 
}

void FastJetAlgo::recoToFastJet(const FastJetAlgo::InputCollection& inputColl) {
  input_.clear();
  typedef InputCollection::const_iterator II;

  unsigned index = 0;
  for(II i=inputColl.begin(); i!=inputColl.end(); ++i, ++index) {
    input_.push_back( PseudoJet( i->px(), i->py(), i->pz(), i->energy() ) );
    input_.back().set_user_index( index );
  }
}

void  FastJetAlgo::runJetClustering() {
  output_.clear();
  if(clusterSequence_) delete clusterSequence_;
  clusterSequence_ = new ClusterSequence(input_, jetDefinition_);

  double ptMin_=2;//COLIN make it an attribute
  output_ = clusterSequence_->inclusive_jets( ptMin_ );
}



const FastJetAlgo::JetCollection& FastJetAlgo::fastJetToReco() {
  
  jetCollection_.clear();

  for(PJI i=output_.begin(); i!=output_.end(); ++i) {
    jetCollection_.push_back( makeJet( *i ) ); 
  }

  return jetCollection_;
}


FastJetAlgo::JetType FastJetAlgo::makeJet( const PseudoJet& pseudoJet) const {
  
  reco::Particle::LorentzVector p4( pseudoJet.px(), 
				    pseudoJet.py(),
				    pseudoJet.pz(),
				    pseudoJet.E() );
  reco::Particle::Point vertex; 
  JetType::Specific specific; 
    
  // need to add the constituents as well (see base Jet, or CompositePtrCandidate)
  reco::Jet::Constituents ptrsToConstituents = makeConstituents( pseudoJet );
  
  makeSpecific( ptrsToConstituents, &specific );
  return JetType(p4, vertex, specific, ptrsToConstituents); 
}
  

reco::Jet::Constituents FastJetAlgo::makeConstituents(const fastjet::PseudoJet& pseudoJet) const {
  
  reco::Jet::Constituents ptrsToConstituents; 
  
  const PseudoJetCollection& constituents 
    = clusterSequence_->constituents(pseudoJet); 
  for(PJI jc=constituents.begin(); jc!=constituents.end(); ++jc) {
    ptrsToConstituents.push_back( edm::Ptr<reco::Candidate>(inputHandle_, jc->user_index() ) ); 
  } 

  return ptrsToConstituents;
}


void FastJetAlgo::printPseudoJets( ostream& out) const {

//   cout<<"Jet Definition:"<<endl;
//   cout<<jetDefinition_;
  cout<<"Pseudo jets:"<<endl;
  unsigned index = 0;
  for(PJI i=output_.begin(); i!=output_.end(); ++i, ++index) {
    cout<<index<<" "<<i->Et()<<endl;
    
    const PseudoJetCollection& constituents = clusterSequence_->constituents( *i );
    for(PJI jc=constituents.begin(); jc!=constituents.end(); ++jc) {
      cout<<"\t"<<jc->user_index()<<" "<<jc->Et()<<endl;
    }
  }
}


void FastJetAlgo::printJets( ostream& out) const {

//   cout<<"Jet Definition:"<<endl;
//   cout<<jetDefinition_;
  cout<<"Jets:"<<endl;
  unsigned index = 0;
  for(JI i=jetCollection_.begin(); i!=jetCollection_.end(); ++i, ++index) {
    cout<<index<<" "<<(*i)<<endl;
    cout<<i->print()<<endl;
  }
}


