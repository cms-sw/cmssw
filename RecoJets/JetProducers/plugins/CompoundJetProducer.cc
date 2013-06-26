
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoJets/JetProducers/plugins/CompoundJetProducer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoJets/JetProducers/interface/JetSpecific.h"

using namespace std;
using namespace reco;
using namespace edm;
using namespace cms;

namespace {
  const bool debug = false;

}


CompoundJetProducer::CompoundJetProducer(edm::ParameterSet const& conf):
  VirtualJetProducer( conf )
{

  produces<reco::BasicJetCollection>();
  // the subjet collections are set through the config file in the "jetCollInstanceName" field.
}


void CompoundJetProducer::inputTowers( ) 
{
  fjCompoundJets_.clear();
  VirtualJetProducer::inputTowers();
}

void CompoundJetProducer::output(edm::Event & iEvent, edm::EventSetup const& iSetup)
{
  // Write jets and constitutents. Will use fjJets_. 
  switch( jetTypeE ) {
  case JetType::CaloJet :
    writeCompoundJets<reco::CaloJet>( iEvent, iSetup );
    break;
  case JetType::PFJet :
    writeCompoundJets<reco::PFJet>( iEvent, iSetup );
    break;
  case JetType::GenJet :
    writeCompoundJets<reco::GenJet>( iEvent, iSetup );
    break;
  case JetType::BasicJet :
    writeCompoundJets<reco::BasicJet>( iEvent, iSetup );
    break;
  default:
    throw cms::Exception("InvalidInput") << "invalid jet type in CompoundJetProducer\n";
    break;
  };

}

/// function template to write out the outputs
template< class T>
void CompoundJetProducer::writeCompoundJets(  edm::Event & iEvent, edm::EventSetup const& iSetup)
{

  // get a list of output jets
  std::auto_ptr<reco::BasicJetCollection>  jetCollection( new reco::BasicJetCollection() );
  // get a list of output subjets
  std::auto_ptr<std::vector<T> >  subjetCollection( new std::vector<T>() );

  // This will store the handle for the subjets after we write them
  edm::OrphanHandle< std::vector<T> > subjetHandleAfterPut;
  // this is the mapping of subjet to hard jet
  std::vector< std::vector<int> > indices;
  // this is the list of hardjet 4-momenta
  std::vector<math::XYZTLorentzVector> p4_hardJets;
  // this is the hardjet areas
  std::vector<double> area_hardJets;


  // Loop over the hard jets
  std::vector<CompoundPseudoJet>::const_iterator it = fjCompoundJets_.begin(),
    iEnd = fjCompoundJets_.end(),
    iBegin = fjCompoundJets_.begin();
  indices.resize( fjCompoundJets_.size() );
  for ( ; it != iEnd; ++it ) {
    int jetIndex = it - iBegin;
    fastjet::PseudoJet localJet = it->hardJet();
    // Get the 4-vector for the hard jet
    p4_hardJets.push_back( math::XYZTLorentzVector(localJet.px(), localJet.py(), localJet.pz(), localJet.e() ));
    area_hardJets.push_back( it->hardJetArea() );

    // create the subjet list
    std::vector<CompoundPseudoSubJet>::const_iterator itSubJetBegin = it->subjets().begin(),
      itSubJet = itSubJetBegin, itSubJetEnd = it->subjets().end();
    for (; itSubJet != itSubJetEnd; ++itSubJet ){

      fastjet::PseudoJet subjet = itSubJet->subjet();
      math::XYZTLorentzVector p4Subjet(subjet.px(), subjet.py(), subjet.pz(), subjet.e() );
      reco::Particle::Point point(0,0,0);

      // This will hold ptr's to the subjets
      std::vector<reco::CandidatePtr> subjetConstituents;

      // Get the transient subjet constituents from fastjet
      std::vector<int> const & subjetFastjetConstituentIndices = itSubJet->constituents();
      std::vector<int>::const_iterator fastSubIt = subjetFastjetConstituentIndices.begin(),
	transConstEnd = subjetFastjetConstituentIndices.end();
      for ( ; fastSubIt != transConstEnd; ++fastSubIt ) {
	// Add a ptr to this constituent
	if ( *fastSubIt < static_cast<int>(inputs_.size()) ) 
	  subjetConstituents.push_back( inputs_[*fastSubIt] );
      }

      // This holds the subjet-to-hardjet mapping
      indices[jetIndex].push_back( subjetCollection->size() );      


      // Add the concrete subjet type to the subjet list to write to event record
      T jet;
      reco::writeSpecific( jet, p4Subjet, point, subjetConstituents, iSetup);
      jet.setJetArea( itSubJet->subjetArea() );
      subjetCollection->push_back( jet );

    }
  }
  // put subjets into event record
  subjetHandleAfterPut = iEvent.put( subjetCollection, jetCollInstanceName_ );
  
  
  // Now create the hard jets with ptr's to the subjets as constituents
  std::vector<math::XYZTLorentzVector>::const_iterator ip4 = p4_hardJets.begin(),
    ip4Begin = p4_hardJets.begin(),
    ip4End = p4_hardJets.end();

  for ( ; ip4 != ip4End; ++ip4 ) {
    int p4_index = ip4 - ip4Begin;
    std::vector<int> & ind = indices[p4_index];
    std::vector<reco::CandidatePtr> i_hardJetConstituents;
    // Add the subjets to the hard jet
    for( std::vector<int>::const_iterator isub = ind.begin();
	 isub != ind.end(); ++isub ) {
      reco::CandidatePtr candPtr( subjetHandleAfterPut, *isub, false );
      i_hardJetConstituents.push_back( candPtr );
    }   
    reco::Particle::Point point(0,0,0);
    reco::BasicJet toput( *ip4, point, i_hardJetConstituents);
    toput.setJetArea( area_hardJets[ip4 - ip4Begin] );
    jetCollection->push_back( toput );
  }
  
  // put hard jets into event record
  iEvent.put( jetCollection);

}
