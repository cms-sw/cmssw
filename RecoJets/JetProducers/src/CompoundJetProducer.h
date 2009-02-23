#ifndef RecoJets_JetProducers_CompoundJetProducer_h
#define RecoJets_JetProducers_CompoundJetProducer_h


/* *********************************************************
  \class CompoundJetProducer

  \brief Jet producer to produce compound jets (i.e. jets of jets)

  \author   Salvatore Rappoccio
  \version  

         Notes on implementation:

	 Because the BaseJetProducer only allows the user to produce
	 one jet collection at a time, this algorithm cannot
	 fit into that paradigm. 

	 All of the "hard" jets are of type BasicJet, since
	 they are "jets of jets". The subjets will be either
	 CaloJets, GenJets, etc.

	 In order to avoid a templatization of the entire
	 EDProducer itself, we only use a templated method
	 to write out the subjets to the event record,
	 and to use that information to write out the
	 hard jets to the event record.

 ************************************************************/


#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "RecoJets/JetAlgorithms/interface/JetMaker.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"


namespace cms
{
  class CompoundJetProducer : public edm::EDProducer
  {
  public:

    CompoundJetProducer(const edm::ParameterSet& ps);

    virtual ~CompoundJetProducer() {}

    //Produces the EDM products
    virtual void produce(edm::Event& e, const edm::EventSetup& c);

    virtual void runAlgorithm( std::vector<fastjet::PseudoJet> const & inputs, 
			       std::vector<CompoundPseudoJet> & outputs,
			       edm::EventSetup const & iSetup ) = 0;
    
  protected:
    edm::InputTag            src_;         /// Input constituents
    std::string              jetType_;     /// Jet type for the subjets
    std::string              subjetColl_;  /// Subjet collection name

  private:
    /// function template to write out the outputs
    template<class T>
    void write_outputs( edm::Event & e, 
			edm::EventSetup const & c,
			std::vector<CompoundPseudoJet> const & outputs,
			edm::Handle< edm::View<reco::Candidate> > const & fInput );
    
    // Overloaded methods to write out specific types
    
    // CaloJet
    void write_specific(reco::CaloJet & jet,
			reco::Particle::LorentzVector const & p4,
			reco::Particle::Point const & point, 
			std::vector<reco::CandidatePtr> const & constituents,
			edm::EventSetup const & c  )
    {
      // Get geometry
      edm::ESHandle<CaloGeometry> geometry;
      c.get<CaloGeometryRecord>().get(geometry);
      const CaloSubdetectorGeometry* towerGeometry = 
	geometry->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId);
      // Switch to IndexedCandidate
      JetReco::InputCollection out_constituents;
      std::vector<reco::CandidatePtr>::const_iterator constituents_begin = constituents.begin(),
	constituents_end = constituents.end(), iconstituent = constituents_begin;
      for ( ; iconstituent != constituents_end; ++iconstituent) {
	out_constituents.push_back( JetReco::InputItem( iconstituent->get(), iconstituent->key() ) );
      }
      // Make the specific
      reco::CaloJet::Specific specific;
      JetMaker::makeSpecific (out_constituents, *towerGeometry, &specific);
      jet = reco::CaloJet( p4, point, specific, constituents);  
    }
    
    // BasicJet
    void write_specific(reco::BasicJet  & jet,
			reco::Particle::LorentzVector const & p4,
			reco::Particle::Point const & point, 
			std::vector<reco::CandidatePtr> const & constituents,
			edm::EventSetup const & c  )
    {
      jet = reco::BasicJet( p4, point, constituents);  
    }
    
    // GenJet
    void write_specific(reco::GenJet  & jet,
			reco::Particle::LorentzVector const & p4,
			reco::Particle::Point const & point, 
			std::vector<reco::CandidatePtr> const & constituents,
			edm::EventSetup const & c  )
    {


      // Switch to IndexedCandidate
      JetReco::InputCollection out_constituents;
      std::vector<reco::CandidatePtr>::const_iterator constituents_begin = constituents.begin(),
	constituents_end = constituents.end(), iconstituent = constituents_begin;
      for ( ; iconstituent != constituents_end; ++iconstituent) {
	out_constituents.push_back( JetReco::InputItem( iconstituent->get(), iconstituent->key() ) );
      }
      // Make the specific
      reco::GenJet::Specific specific;
      JetMaker::makeSpecific (out_constituents, &specific);
      jet = reco::GenJet( p4, point, specific, constituents);  
    }
    
    // PFJet
    void write_specific(reco::PFJet  & jet,
			reco::Particle::LorentzVector const & p4,
			reco::Particle::Point const & point, 
			std::vector<reco::CandidatePtr> const & constituents,
			edm::EventSetup const & c  )
    {
      // Switch to IndexedCandidate
      JetReco::InputCollection out_constituents;
      std::vector<reco::CandidatePtr>::const_iterator constituents_begin = constituents.begin(),
	constituents_end = constituents.end(), iconstituent = constituents_begin;
      for ( ; iconstituent != constituents_end; ++iconstituent) {
	out_constituents.push_back( JetReco::InputItem( iconstituent->get(), iconstituent->key() ) );
      }
      // Make the specific
      reco::PFJet::Specific specific;
      JetMaker::makeSpecific (out_constituents, &specific);
      jet = reco::PFJet( p4, point, specific, constituents);  
    }
    
    

  };



/// function template to write out the outputs
template< class T>
void CompoundJetProducer::write_outputs( edm::Event & e, 
				      edm::EventSetup const & c,
				      std::vector<CompoundPseudoJet> const & outputs,
				      edm::Handle< edm::View<reco::Candidate> > const & fInputHandle )
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

  // Loop over the hard jets
  std::vector<CompoundPseudoJet>::const_iterator it = outputs.begin(),
    iEnd = outputs.end(),
    iBegin = outputs.begin();
  indices.resize( outputs.size() );
  for ( ; it != iEnd; ++it ) {
    int jetIndex = it - iBegin;
    fastjet::PseudoJet localJet = it->hardJet();
    // Get the 4-vector for the hard jet
    p4_hardJets.push_back( math::XYZTLorentzVector(localJet.px(), localJet.py(), localJet.pz(), localJet.e() ));

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
	transConstBegin = subjetFastjetConstituentIndices.begin(),
	transConstEnd = subjetFastjetConstituentIndices.end();
      for ( ; fastSubIt != transConstEnd; ++fastSubIt ) {
	// Add a ptr to this constituent
	if ( *fastSubIt < static_cast<int>(fInputHandle->size()) ) 
	  subjetConstituents.push_back(reco::CandidatePtr(fInputHandle, *fastSubIt));
      }

      // This holds the subjet-to-hardjet mapping
      indices[jetIndex].push_back( subjetCollection->size() );      


      // Add the concrete subjet type to the subjet list to write to event record
      T jet;
      write_specific( jet, p4Subjet, point, subjetConstituents, c);
      subjetCollection->push_back( jet );

    }
  }
  // put subjets into event record
  subjetHandleAfterPut = e.put( subjetCollection, subjetColl_ );
  
  
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
    jetCollection->push_back( reco::BasicJet( *ip4, point, i_hardJetConstituents) );
  }
  

  // put hard jets into event record
  e.put( jetCollection);

}




  
}


#endif
