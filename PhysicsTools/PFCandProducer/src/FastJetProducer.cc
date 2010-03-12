#include "PhysicsTools/PFCandProducer/interface/FastJetProducer.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "fastjet/ClusterSequence.hh"


using namespace pf2pat;
using namespace fastjet;
using namespace std;

FastJetProducer::FastJetProducer( const edm::ParameterSet& ps ) 
  : clusterSequence_(0) {
  setJetDefinition( ps ); 
}

void FastJetProducer::setJetDefinition( const edm::ParameterSet& ps) {
  // here extract parameter set info to make the jet definition
  JetDefinition jetDef( kt_algorithm, 0.5);
  setJetDefinition( jetDef ); 
}

const FastJetProducer::JetCollection& FastJetProducer::produce( const FastJetProducer::InputCollection& inputColl) {

  recoToFastJet( inputColl );
  runJetClustering();
  return fastJetToReco(); 
}

void FastJetProducer::recoToFastJet(const FastJetProducer::InputCollection& inputColl) {
  input_.clear();
  typedef InputCollection::const_iterator II;

  unsigned index = 0;
  for(II i=inputColl.begin(); i!=inputColl.end(); ++i, ++index) {
    input_.push_back( PseudoJet( i->px(), i->py(), i->pz(), i->energy() ) );
    input_.back().set_user_index( index );
  }
}

void  FastJetProducer::runJetClustering() {
  output_.clear();
  if(clusterSequence_) delete clusterSequence_;
  clusterSequence_ = new ClusterSequence(input_, jetDefinition_);
  output_ = clusterSequence_->inclusive_jets();
}

const  FastJetProducer::JetCollection&  FastJetProducer::fastJetToReco() {
  jetCollection_.clear();
  
  return jetCollection_;
}


void FastJetProducer::printPseudoJets( ostream& out) const {
  typedef PseudoJetCollection::const_iterator PJI;

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
