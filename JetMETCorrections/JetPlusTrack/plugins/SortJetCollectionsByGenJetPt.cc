#include "JetMETCorrections/JetPlusTrack/plugins/SortJetCollectionsByGenJetPt.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
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
  : src_( pset.getParameter<edm::InputTag>("src") ),
    matched_( pset.getParameter<edm::InputTag>("matched") ),
    jets_( pset.getParameter< std::vector<edm::InputTag> >("jets") )
{
  
  // reco::GenJet products
  produces< std::vector<reco::GenJet> >( matched_.label() );
  
  // reco::BasicJet products
  std::vector<edm::InputTag>::const_iterator ii = jets_.begin();
  std::vector<edm::InputTag>::const_iterator jj = jets_.end();
  for ( ; ii != jj; ++ii ) {
    if ( ii->label().empty() ) { continue; }
    produces< std::vector<reco::BasicJet> >( ii->label() );
  }
  
}

// -----------------------------------------------------------------------------
//
jpt::SortJetCollectionsByGenJetPt::~SortJetCollectionsByGenJetPt() {;}

// -----------------------------------------------------------------------------
//
void jpt::SortJetCollectionsByGenJetPt::produce( edm::Event& event, 
						 const edm::EventSetup& setup ) {


  // ---------- Create sorted map ----------

  
  // Container for matched jets
  std::map<reco::GenJet,reco::Jet> matched_map;
  
  // Retrieve matched reco::GenJets
  edm::Handle< edm::Association< std::vector<reco::GenJet> > > matched;
  if ( !matched_.label().empty() ) { event.getByLabel( matched_, matched ); }

  // Retrieve matched reco::CaloJets
  edm::Handle< edm::View<reco::Jet> > src;
  event.getByLabel( src_, src );
    
  // Iterate through reco::CaloJets
  edm::View<reco::Jet>::const_iterator isrc = src->begin();
  edm::View<reco::Jet>::const_iterator jsrc = src->end();
  for ( ; isrc != jsrc; ++isrc ) {
    
    // Create edm::Refs
    uint32_t index = isrc - src->begin();
    edm::RefToBase<reco::Jet> base_ref = src->refAt(index);
    edm::Ref< std::vector<reco::GenJet> > gen_ref = (*matched)[base_ref];
      
    // Store in map
    if ( base_ref.isNonnull() && 
	 base_ref.isAvailable() &&
	 gen_ref.isNonnull() && 
	 gen_ref.isAvailable() ) {
      matched_map.insert( std::make_pair( *gen_ref, *base_ref ) );
    }
      
  }


  // ---------- Print sorted map ----------


  if ( edm::isDebugEnabled() ) { 
    std::stringstream ss;
    ss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
       << " MatchMap has " << matched_map.size() 
       << " entries: " << std::endl
       << "    GenPt    JetPt " << std::endl;
    std::map<reco::GenJet,reco::Jet>::const_iterator ii = matched_map.begin();
    std::map<reco::GenJet,reco::Jet>::const_iterator jj = matched_map.end();
    for ( ; ii != jj; ++ii ) {
      ss << " "
	 << std::setfill(' ') << std::setw(8) << ii->first.pt() << " "
	 << std::setfill(' ') << std::setw(8) << ii->second.pt() << " "
	 << std::endl;
    }
    LogTrace("TEST") << ss.str();
  }


  // ---------- Create vectors of reco::GenJets and "reference" reco::Jets ----------


  std::vector<reco::GenJet> gen_jets;
  std::vector<reco::Jet> ref_jets;

  uint32_t index = 0;
  std::stringstream ss;
  if ( edm::isDebugEnabled() ) {
    ss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
       << " Ordered vectors have " << matched_map.size() 
       << " jets:" << std::endl
       << " GenJetPt  (#) : RefJetPt  (#)" << std::endl;
  }
  std::map<reco::GenJet,reco::Jet>::const_iterator imap = matched_map.begin();
  std::map<reco::GenJet,reco::Jet>::const_iterator jmap = matched_map.end();
  for ( ; imap != jmap; ++imap ) {
    if ( imap->first.pt() > 10. ) {
      gen_jets.push_back( imap->first );
      ref_jets.push_back( imap->second );
      if ( edm::isDebugEnabled() ) {
	std::stringstream temp; temp << "(" << uint32_t(index) << ")";
	ss << " "
	   << std::setfill(' ') << std::setw(8) << imap->first.pt() << " "
	   << std::setfill(' ') << std::setw(4) << temp.str() << " : "
	   << std::setfill(' ') << std::setw(8) << imap->second.pt() << " "
	   << std::setfill(' ') << std::setw(4) << temp.str() << std::endl;
      }
    }
    index++;
  }
  if ( edm::isDebugEnabled() ) { LogTrace("TEST") << ss.str(); }

  // Put reco::GenJets in Event
  std::auto_ptr< std::vector<reco::GenJet> > ordered( new std::vector<reco::GenJet>(gen_jets) );
  event.put( ordered, matched_.label() );
  

  // ---------- Create reco::BasicJet products ----------


  {
    std::vector<edm::InputTag>::const_iterator ijet = jets_.begin();
    std::vector<edm::InputTag>::const_iterator jjet = jets_.end();
    for ( ; ijet != jjet; ++ijet ) {
      
      // Check for non-null label
      if ( ijet->label().empty() ) { continue; }

      // Debug
      std::stringstream ss;

      // Retrieve reco::Jet collection 
      edm::Handle< edm::View<reco::Jet> > jets;
      event.getByLabel( *ijet, jets );
      
      // Sort reco::Jet collection according to GenJet pt
      std::auto_ptr< std::vector<reco::BasicJet> > ordered_jets( new std::vector<reco::BasicJet>() );
      std::vector<reco::Jet>::const_iterator iref = ref_jets.begin();
      std::vector<reco::Jet>::const_iterator jref = ref_jets.end();
      for ( ; iref != jref; ++iref ) {
	edm::View<reco::Jet>::const_iterator iter = std::find( jets->begin(), jets->end(), *iref );
	double jet_pt = -1.;
	int32_t index = -1;
	if ( iter != jets->end() ) { 
	  reco::BasicJet jet( iter->p4(), iter->vertex(), iter->getJetConstituents() );
	  ordered_jets->push_back( jet ); 
	  jet_pt = iter->pt();
	  index = static_cast<int32_t>( iter - jets->begin() );
	}
	if ( edm::isDebugEnabled() ) {
	  std::stringstream temp1; temp1 << "(" << static_cast<uint32_t>( iref - ref_jets.begin() ) << ")";
	  std::stringstream temp2; temp2 << "(" << index << ")";
	  ss << " "
	     << std::setfill(' ') << std::setw(8) << iref->pt() << " "
	     << std::setfill(' ') << std::setw(4) << temp1.str() << " :  "
	     << std::setfill(' ') << std::setw(8) << jet_pt << " "
	     << std::setfill(' ') << std::setw(4) << temp2.str() << std::endl;
	}
      }

      if ( edm::isDebugEnabled() ) { 
	std::stringstream sss;
	sss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
	    << " Kept reco::BasicJets (" << ordered_jets->size() 
	    << ") for \"" << ijet->label() << "\"." << std::endl
	    << " RefJetPt  (#) : CaloJetPt  (#)" << std::endl
	    << ss.str();
	LogTrace("TEST") << sss.str(); 
      }

      // Put reco::BasicJets in Event
      event.put( ordered_jets, ijet->label() );

    }
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace jpt;
DEFINE_FWK_MODULE(SortJetCollectionsByGenJetPt);
