#include "JetMETCorrections/JetPlusTrack/plugins/SortJetCollectionsByGenJetPt.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
//#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iomanip>
#include <map>

using namespace reco;

// -----------------------------------------------------------------------------
//
jpt::SortJetCollectionsByGenJetPt::SortJetCollectionsByGenJetPt( const edm::ParameterSet& pset ) 
  : matchedCaloJets_( pset.getParameter<edm::InputTag>("matchedCaloJets") ),
    matchedGenJets_( pset.getParameter<edm::InputTag>("matchedGenJets") ),
    //genJets_( pset.getParameter<edm::InputTag>("genJets") ),
    caloJets_( pset.getParameter< std::vector<edm::InputTag> >("caloJets") ),
    patJets_( pset.getParameter< std::vector<edm::InputTag> >("patJets") )
{

  // reco::GenJet products
  produces< std::vector<reco::GenJet> >( genJets_.label() );

  // reco::CaloJet products
  {
    std::vector<edm::InputTag>::const_iterator ii = caloJets_.begin();
    std::vector<edm::InputTag>::const_iterator jj = caloJets_.end();
    for ( ; ii != jj; ++ii ) {
      if ( ii->label().empty() ) { continue; }
      produces<CaloJetCollection>( ii->label() );
    }
  }

  // pat::Jet products
  { 
    std::vector<edm::InputTag>::const_iterator ii = patJets_.begin();
    std::vector<edm::InputTag>::const_iterator jj = patJets_.end();
    for ( ; ii != jj; ++ii ) {
      if ( ii->label().empty() ) { continue; }
      produces< std::vector<pat::Jet> >( ii->label() );
    }
  }

}

// -----------------------------------------------------------------------------
//
jpt::SortJetCollectionsByGenJetPt::~SortJetCollectionsByGenJetPt() {;}

// -----------------------------------------------------------------------------
//
void jpt::SortJetCollectionsByGenJetPt::produce( edm::Event& event, const edm::EventSetup& setup ) {

  // -------------------- Establish matching --------------------
  
  // Container for matched jets
  std::map<reco::GenJet,reco::Jet> match_map;
  
  {

    // Retrieve matched reco::GenJets
    edm::Handle< edm::Association< std::vector<reco::GenJet> > > matched_gen_jets;
    if ( !matchedGenJets_.label().empty() ) { event.getByLabel( matchedGenJets_, matched_gen_jets ); }

    // Retrieve matched reco::CaloJets
    edm::Handle< edm::View<reco::Jet> > matched_jets;
    event.getByLabel( matchedCaloJets_, matched_jets );
    
    // Iterate through reco::CaloJets
    edm::View<reco::Jet>::const_iterator ii = matched_jets->begin();
    edm::View<reco::Jet>::const_iterator jj = matched_jets->end();
    for ( ; ii != jj; ++ii ) {
    
      // Create edm::Refs
      uint32_t index = ii - matched_jets->begin();
      edm::RefToBase<reco::Jet> base_ref = matched_jets->refAt(index);
      edm::Ref< std::vector<reco::GenJet> > gen_ref = (*matched_gen_jets)[base_ref];
      
      // Store in map
      if ( base_ref.isNonnull() && 
	   base_ref.isAvailable() &&
	   gen_ref.isNonnull() && 
	   gen_ref.isAvailable() ) {
	match_map.insert( std::make_pair( *gen_ref, *base_ref ) );
      }
      
    }
    
    std::stringstream ss;
    ss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
       << " MatchMap has " << match_map.size() << " entries: " 
       << std::endl
       << "    GenPt    JetPt " << std::endl;
    std::map<reco::GenJet,reco::Jet>::const_iterator iii = match_map.begin();
    std::map<reco::GenJet,reco::Jet>::const_iterator jjj = match_map.end();
    for ( ; iii != jjj; ++iii ) {
      ss << " "
	 << std::setfill(' ') << std::setw(8) << iii->first.pt() << " "
	 << std::setfill(' ') << std::setw(8) << iii->second.pt() << " "
	 << std::endl;
    }
    LogTrace("TEST") << ss.str();
    
  }
  
  // ---------- Create reco::GenJet and reference reco::Jet collections ----------

  std::vector<reco::GenJet> gen_jets;
  std::vector<reco::Jet> ref_jets;
  {
    uint32_t index = 0;
    std::stringstream ss;
    if ( edm::isDebugEnabled() ) {
      ss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
	 << " ORDERED jets (" << match_map.size() 
	 << ")." << std::endl
	 << " GenJetPt  (#) : RefJetPt  (#)" << std::endl;
    }
    std::map<reco::GenJet,reco::Jet>::const_iterator ii = match_map.begin();
    std::map<reco::GenJet,reco::Jet>::const_iterator jj = match_map.end();
    for ( ; ii != jj; ++ii ) {
      if ( ii->first.pt() > 10. ) {
	gen_jets.push_back( ii->first );
	ref_jets.push_back( ii->second );
	if ( edm::isDebugEnabled() ) {
	  std::stringstream temp; temp << "(" << uint32_t(index) << ")";
	  ss << " "
	     << std::setfill(' ') << std::setw(8) << ii->first.pt() << " "
	     << std::setfill(' ') << std::setw(4) << temp.str() << " : "
	     << std::setfill(' ') << std::setw(8) << ii->second.pt() << " "
	     << std::setfill(' ') << std::setw(4) << temp.str() << std::endl;
	}
      }
      index++;
    }
    if ( edm::isDebugEnabled() ) { LogTrace("TEST") << ss.str(); }

    std::auto_ptr< std::vector<reco::GenJet> > ordered( new std::vector<reco::GenJet>(gen_jets) );
    event.put(ordered,genJets_.label());
    
  }
  
  // ---------- Create reco::CaloJet products ----------

  {
    std::vector<edm::InputTag>::const_iterator ii = caloJets_.begin();
    std::vector<edm::InputTag>::const_iterator jj = caloJets_.end();
    for ( ; ii != jj; ++ii ) {
      
      // Check for non-null label
      if ( ii->label().empty() ) { continue; }

      // Debug
      std::stringstream ss;

      // Retrieve reco::CaloJet collection 
      edm::Handle< edm::View<reco::CaloJet> > jets;
      event.getByLabel( *ii, jets );

      // Sort reco::CaloJet collection according to GenJet pt
      std::auto_ptr<CaloJetCollection> ordered( new CaloJetCollection() );
      std::vector<reco::Jet>::const_iterator iii = ref_jets.begin();
      std::vector<reco::Jet>::const_iterator jjj = ref_jets.end();
      for ( ; iii != jjj; ++iii ) {
	edm::View<reco::CaloJet>::const_iterator iter = std::find( jets->begin(), jets->end(), *iii );
	double jet_pt = -1.;
	int32_t index = -1;
	if ( iter != jets->end() ) { 
	  ordered->push_back(*iter); 
	  jet_pt = iter->pt();
	  index = static_cast<int32_t>( iter - jets->begin() );
	}
	if ( edm::isDebugEnabled() ) {
	  std::stringstream temp1; temp1 << "(" << static_cast<uint32_t>( iii - ref_jets.begin() ) << ")";
	  std::stringstream temp2; temp2 << "(" << index << ")";
	  ss << " "
	     << std::setfill(' ') << std::setw(8) << iii->pt() << " "
	     << std::setfill(' ') << std::setw(4) << temp1.str() << " :  "
	     << std::setfill(' ') << std::setw(8) << jet_pt << " "
	     << std::setfill(' ') << std::setw(4) << temp2.str() << std::endl;
	}
      }

      if ( edm::isDebugEnabled() ) { 
	std::stringstream sss;
	sss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
	    << " KEPT reco::CaloJets (" << ordered->size() 
	    << ") for \"" << ii->label() << "\"." << std::endl
	    << " RefJetPt  (#) : CaloJetPt  (#)" << std::endl
	    << ss.str();
	LogTrace("TEST") << sss.str(); 
      }

      event.put(ordered,ii->label());

    }
  }
  
//   // ---------- Create reco::PatJet products ----------

//   {
//     std::vector<edm::InputTag>::const_iterator ii = patJets_.begin();
//     std::vector<edm::InputTag>::const_iterator jj = patJets_.end();
//     for ( ; ii != jj; ++ii ) {

//       // Check for non-null label
//       if ( ii->label().empty() ) { continue; }

//       // Debug
//       std::stringstream ss;

//       // Retrieve pat::Jet collection 
//       edm::Handle< edm::View<pat::Jet> > jets;
//       event.getByLabel( *ii, jets );

//       // Sort pat::Jet collection according to GenJet pt
//       std::auto_ptr< std::vector<pat::Jet> > ordered( new std::vector<pat::Jet>() );
//       std::vector<pat::Jet>::const_iterator iii = ref_jets.begin();
//       std::vector<pat::Jet>::const_iterator jjj = ref_jets.end();
//       for ( ; iii != jjj; ++iii ) {
// 	reco::CaloJet temp( iii->p4(), iii->caloSpecific(), iii->getJetConstituents() );

// 	edm::View<pat::Jet>::const_iterator iter = std::find( jets->begin(), jets->end(), *iii );
// 	double jet_pt = -1.;
// 	int32_t index = -1;
// 	if ( iter != jets->end() ) { 
// 	  ordered->push_back(*iter); 
// 	  jet_pt = iter->pt();
// 	  index = static_cast<int32_t>( iter - jets->begin() );
// 	}
// 	if ( edm::isDebugEnabled() ) {
// 	  std::stringstream temp1; temp1 << "(" << static_cast<uint32_t>( iii - ref_jets.begin() ) << ")";
// 	  std::stringstream temp2; temp2 << "(" << index << ")";
// 	  ss << " "
// 	     << std::setfill(' ') << std::setw(8) << iii->pt() << " "
// 	     << std::setfill(' ') << std::setw(4) << temp1.str() << " : "
// 	     << std::setfill(' ') << std::setw(8) << jet_pt << " "
// 	     << std::setfill(' ') << std::setw(4) << temp2.str() << std::endl;
// 	}
//       }

//       if ( edm::isDebugEnabled() ) { 
// 	std::stringstream sss;
// 	sss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
// 	    << " KEPT pat::Jets (" << ordered->size() 
// 	    << ") for \"" << ii->label() << "\"." << std::endl
// 	    << " RefJetPt  (#) : PatJetPt  (#)" << std::endl
// 	    << ss.str();
// 	LogTrace("TEST") << sss.str(); 
//       }

//       event.put(ordered,ii->label());
      
//     }
//   }

}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace jpt;
DEFINE_FWK_MODULE(SortJetCollectionsByGenJetPt);
