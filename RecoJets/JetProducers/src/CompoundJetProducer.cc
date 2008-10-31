
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CompoundJetProducer.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace reco;
using namespace JetReco;
using namespace edm;
using namespace cms;

namespace {
  const bool debug = false;

}


CompoundJetProducer::CompoundJetProducer(edm::ParameterSet const& conf):
  src_       (conf.getParameter<edm::InputTag>("src")),              // input collection
  jetType_   (conf.getUntrackedParameter<std::string>  ("jetType")), // jet reconstruction type
  subjetColl_(conf.getParameter<std::string>("subjetColl"))          // subjet collection name

{
  if ( jetType_ == "CaloJet" )
    produces<reco::CaloJetCollection>(subjetColl_).setBranchAlias(subjetColl_);
  else if ( jetType_ == "GenJet" )
    produces<reco::GenJetCollection>(subjetColl_).setBranchAlias(subjetColl_);
  else if ( jetType_ == "BasicJet" )
    produces<reco::BasicJetCollection>(subjetColl_).setBranchAlias(subjetColl_);
  else if ( jetType_ == "PFJet" )
    produces<reco::PFJetCollection>(subjetColl_).setBranchAlias(subjetColl_);
  else {
    throw cms::Exception("Invalid input type") << "Input type for CompoundJetProducer is invalid\n";
  }


  produces<reco::BasicJetCollection>();
}
  
void CompoundJetProducer::produce( edm::Event & e, const edm::EventSetup & c ) 
{

  bool verbose = false;

 
  // -------------------------------------------------------
  // Set up the constituent list
  // -------------------------------------------------------

  // list of fastjet pseudojet constituents
  vector<fastjet::PseudoJet> inputs;


  // get input from event record
  Handle<View<Candidate> > fInputHandle;
  e.getByLabel( src_, fInputHandle );

  View<Candidate> const & fInput = *fInputHandle;

  // Fill list of fastjet pseudojet constituents
  View<Candidate>::const_iterator inputIt = fInput.begin(),
    inputEnd = fInput.end(),
    inputBegin = inputIt;
  if ( verbose ) cout << "Adding cell particles, n = " << fInput.size()  << endl;

  for ( ; inputIt != inputEnd; ++inputIt ) {
    inputs.push_back (fastjet::PseudoJet (inputIt->px(),inputIt->py(),inputIt->pz(),inputIt->energy()));
    inputs.back().set_user_index(inputIt - inputBegin);
  }

  // Sort pseudojets by et
  GreaterByEtPseudoJet compEt;
  sort( inputs.begin(), inputs.end(), compEt );

  // Here is the list of pseudojet "hard + soft" jets in a structure
  vector<CompoundPseudoJet> outputs;
  



  // -------------------------------------------------------
  // Run the actual algorithm  
  // -------------------------------------------------------
  runAlgorithm( inputs, outputs, c );

  // -------------------------------------------------------
  // Now fill the outputs
  // -------------------------------------------------------

  if ( jetType_ == "CaloJet" )
    write_outputs<reco::CaloJet>( e, c, outputs, fInputHandle ); 
  else if ( jetType_ == "GenJet" )
    write_outputs<reco::GenJet>( e, c, outputs, fInputHandle ); 
  else if ( jetType_ == "BasicJet" )
    write_outputs<reco::BasicJet>( e, c, outputs, fInputHandle ); 
  else if ( jetType_ == "PFJet" )
    write_outputs<reco::PFJet>( e, c, outputs, fInputHandle ); 




}


