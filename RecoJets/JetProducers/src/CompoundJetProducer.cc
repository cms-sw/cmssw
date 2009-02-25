
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
  subjetColl_(conf.getParameter<std::string>("subjetColl")),         // subjet collection name
  mEtInputCut (conf.getParameter<double>("inputEtMin")),
  mEInputCut (conf.getParameter<double>("inputEMin")),
  mVertexCorrectedInput(false),
  maxBadEcalCells        (9999999),
  maxRecoveredEcalCells  (9999999),
  maxProblematicEcalCells(9999999),
  maxBadHcalCells        (9999999),
  maxRecoveredHcalCells  (9999999),
  maxProblematicHcalCells(9999999)
{
  if ( jetType_ == "CaloJet" ) {
    mVertexCorrectedInput=conf.getParameter<bool>("correctInputToSignalVertex");    
    if (mVertexCorrectedInput){
      mPVCollection=conf.getParameter<edm::InputTag>("pvCollection");
    }
    // Add anomalous cell cuts
    maxBadEcalCells         = conf.getParameter<unsigned int>("maxBadEcalCells");
    maxRecoveredEcalCells   = conf.getParameter<unsigned int>("maxRecoveredEcalCells");
    maxProblematicEcalCells = conf.getParameter<unsigned int>("maxProblematicEcalCells");
    maxBadHcalCells         = conf.getParameter<unsigned int>("maxBadHcalCells");
    maxRecoveredHcalCells   = conf.getParameter<unsigned int>("maxRecoveredHcalCells");
    maxProblematicHcalCells = conf.getParameter<unsigned int>("maxProblematicHcalCells");
    produces<reco::CaloJetCollection>(subjetColl_).setBranchAlias(subjetColl_);
  }
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

 
  //set default vertex for undefined cases:
  vertex=reco::Jet::Point(0,0,0);
  //getSignalVertex (when producing caloJets, and configuration wants it)
  if (jetType_ == "CaloJet") {
    if (mVertexCorrectedInput){
      edm::Handle<reco::VertexCollection> thePrimaryVertexCollection;
      e.getByLabel(mPVCollection,thePrimaryVertexCollection);
      if ((*thePrimaryVertexCollection).size()>0){
	vertex = (*thePrimaryVertexCollection)[0].position();
      }
      // no else needed, vertex already set to (0,0,0). 
    }
  }
  
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

  vector<Candidate*> garbageCollection;
  if (mVertexCorrectedInput) garbageCollection.reserve(fInputHandle->size());
  for ( ; inputIt != inputEnd; ++inputIt ) {
    JetReco::InputItem tmpInput(&(*inputIt),inputIt - inputBegin);
    if (mVertexCorrectedInput){
      tmpInput.setOriginal(tmpInput.get());
      const CaloTower *tower=dynamic_cast<const CaloTower*>(&(*inputIt));
      Candidate* tmpCandidate=new CaloTower(*tower);
      math::PtEtaPhiMLorentzVector newCaloTowerVector(tower->p4(vertex));
      reco::Particle::LorentzVector correctedP4(newCaloTowerVector.px(),newCaloTowerVector.py(),newCaloTowerVector.pz(),newCaloTowerVector.energy());
      garbageCollection.push_back(tmpCandidate);
      tmpCandidate->setP4(correctedP4);
      tmpInput.setBase(tmpCandidate);
    }

    // Make cuts on input towers
    if (
	// 4-vector cuts
	(mEtInputCut <= 0 || tmpInput->et() > mEtInputCut) &&
	(mEInputCut <= 0 || tmpInput->energy() > mEInputCut)) {
      // Anomalous cell cuts if this is CaloTower input
      const CaloTower * tower = dynamic_cast<const CaloTower *>(&(*inputIt));
      if (
	  // If this is a calo tower, make cuts on anomalous cells
	  (tower != 0  && 
	   tower->numBadEcalCells() <= maxBadEcalCells &&
	   tower->numRecoveredEcalCells() <= maxRecoveredEcalCells &&
	   tower->numProblematicEcalCells() <= maxProblematicEcalCells &&
	   tower->numBadHcalCells() <= maxBadHcalCells &&
	   tower->numRecoveredHcalCells() <= maxRecoveredHcalCells &&
	   tower->numProblematicHcalCells() <= maxProblematicHcalCells) 
	  ||
	  // If this isn't a calo tower, just pass it
	  tower == 0
	  ) {
	
	inputs.push_back (fastjet::PseudoJet (tmpInput->px(),tmpInput->py(),tmpInput->pz(),tmpInput->energy()));
	inputs.back().set_user_index(inputIt - inputBegin);
      }
    }
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


  //clean up garbage from modified calojet input:
  if (mVertexCorrectedInput){
    for (vector<Candidate*>::iterator iter=garbageCollection.begin();iter!=garbageCollection.end();++iter){
      delete *iter;
    }
    
  }


}


