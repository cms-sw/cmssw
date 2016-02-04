#include "RecoJets/JetAlgorithms/interface/SubJetAlgorithm.h"
#include "PrunedRecombPlugin.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"



using namespace std;
using namespace edm;


bool SubJetAlgorithm::get_pruning() const{
    return enable_pruning_;
}

void SubJetAlgorithm::set_zcut(double z){
    zcut_ = z;
}

void SubJetAlgorithm::set_rcut_factor(double r){
    rcut_factor_ = r;
}


//  Run the algorithm
//  ------------------
void SubJetAlgorithm::run( const vector<fastjet::PseudoJet> & cell_particles,
			     vector<CompoundPseudoJet> & hardjetsOutput,
			     edm::EventSetup const & c ) {
  fastjet::Strategy strategy = fastjet::Best;
  fastjet::RecombinationScheme recombScheme = fastjet::E_scheme;
  fastjet::JetAlgorithm algorithm = static_cast<fastjet::JetAlgorithm>(algorithm_);
  //jet definition, as given in the configuration file, jetDef0:
  fastjet::JetDefinition jetDef0(algorithm, jetsize_, recombScheme, strategy);

  //for actual jet clustering, either the pruned or the original version is used.
  //For the pruned version, a new jet definition using the PrunedRecombPlugin is required:
  std::auto_ptr<fastjet::JetDefinition> pjetdef;
  std::auto_ptr<fastjet::PrunedRecombPlugin> PRplugin;
  if(enable_pruning_){
      PRplugin.reset(new fastjet::PrunedRecombPlugin(jetDef0, jetDef0, zcut_, rcut_factor_));
      pjetdef.reset(new fastjet::JetDefinition(PRplugin.get()));
  }

  //the jet definition which is actually used to cluster the jet is "jetDef". Either
  // the pruned definition or the "original", unpruned one.
  const fastjet::JetDefinition & jetDef = enable_pruning_?(*pjetdef):jetDef0;

  //cluster the jets with the jet definition jetDef:
  fastjet::ClusterSequence clusterSeq(cell_particles, jetDef);
  vector<fastjet::PseudoJet> inclusiveJets = clusterSeq.inclusive_jets(ptMin_);

  // These will store the indices of each subjet that 
  // are present in each jet
  vector<vector<int> > indices(inclusiveJets.size());
  // Loop over inclusive jets, attempt to find substructure
  vector<fastjet::PseudoJet>::iterator jetIt = inclusiveJets.begin();
  for ( ; jetIt != inclusiveJets.end(); ++jetIt ) {
    //decompose into requested number of subjets:
    vector<fastjet::PseudoJet> subjets = clusterSeq.exclusive_subjets(*jetIt, nSubjets_);
    //create the subjets objects to put into the "output" objects
    vector<CompoundPseudoSubJet>  subjetsOutput;
    std::vector<fastjet::PseudoJet>::const_iterator itSubJetBegin = subjets.begin(),
      itSubJet = itSubJetBegin, itSubJetEnd = subjets.end();
    for (; itSubJet != itSubJetEnd; ++itSubJet ){
      // Get the transient subjet constituents from fastjet
      vector<fastjet::PseudoJet> subjetFastjetConstituents = clusterSeq.constituents( *itSubJet );
      // Get the indices of the constituents:
      vector<int> constituents;
      vector<fastjet::PseudoJet>::const_iterator fastSubIt = subjetFastjetConstituents.begin(),
	transConstBegin = subjetFastjetConstituents.begin(),
	transConstEnd = subjetFastjetConstituents.end();
      for ( ; fastSubIt != transConstEnd; ++fastSubIt ) {
	if (fastSubIt->user_index() >= 0) {
	  constituents.push_back( fastSubIt->user_index() );
	}
      }
      // Make a CompoundPseudoSubJet object to hold this subjet and the indices of its constituents
      subjetsOutput.push_back( CompoundPseudoSubJet( *itSubJet, constituents ) );
    }
    // Make a CompoundPseudoJet object to hold this hard jet, and the subjets that make it up
    hardjetsOutput.push_back( CompoundPseudoJet( *jetIt, subjetsOutput ) );
  }
}
