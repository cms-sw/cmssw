#include "JetMETCorrections/JetPlusTrack/plugins/SortJetCollectionsByGenJetPt.h"
#include "DataFormats/Common/interface/View.h"
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
  : //matchedJets_( pset.getParameter<edm::InputTag>("matchedJets") ),
    genJets_( pset.getParameter<edm::InputTag>("genJets") ),
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
      produces< std::vector<reco::CaloJet> >( ii->label() );
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


  // Retrieve pat::Jet collection used to access matched reco::GenJets
  edm::Handle< edm::View<pat::Jet> > temp_jets;
  event.getByLabel( genJets_, temp_jets );
  
  // Container for mapping
  std::map<GenJet,pat::Jet> mapping;

//    // MC matching
//    edm::Handle< edm::Association< reco::GenParticleCollection > > matched_jets;
//    if ( !matchedJets_.label().empty() ) { event.getByLabel( matchedJets_, matched_jets ); }
  
//    // Embed GenParticle
//    if ( !matchedJets_.label().empty() ) {
//      uint32_t index = iphoton - new_photons->begin();
//      edm::RefToBase<pat::Photon> photon_ref = orig_photons->refAt(index);
//      if ( photon_ref.isNonnull() ) { 
//        reco::GenParticleRef gen = (*matched_jets)[photon_ref];
//        if ( gen.isNonnull() ) { 1
//  	iphoton->addGenParticleRef(gen);
//  	iphoton->embedGenParticle();
//        } //else { std::cout << "NULL Ref" << std::endl; }
//      } //else { std::cout << "NULL RefToBase" << std::endl; }
//    }

  // ---------- Extract reco::GenJets ---------- 
  
  { 
    std::stringstream ss;
    if ( edm::isDebugEnabled() ) {
      ss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
	 << " UNORDERED jets (" << temp_jets->size() 
	 << ")." << std::endl
	 << " PatJetPt  (#) : GenJetPt  (#)" << std::endl;
    }
    int32_t index = 0;
    edm::View<pat::Jet>::const_iterator ii = temp_jets->begin();
    edm::View<pat::Jet>::const_iterator jj = temp_jets->end();
    for ( ; ii != jj; ++ii ) {
      double jet_pt = -1.;
      if ( ii->genJet() && ii->genJet()->pt() > 10. ) { 
	mapping.insert( std::make_pair(*(ii->genJet()),*ii) ); 
	jet_pt = ii->genJet()->pt();
	index++;
      } 
      if ( edm::isDebugEnabled() ) {
	std::stringstream temp1; temp1 << "(" << static_cast<uint32_t>( ii - temp_jets->begin() ) << ")";
	std::stringstream temp2; temp2 << "(" << ( jet_pt < 0. ? -1 : index-1 ) << ")";
	ss << " "
	   << std::setfill(' ') << std::setw(8) << ii->pt() << " "
	   << std::setfill(' ') << std::setw(4) << temp1.str() << " : "
	   << std::setfill(' ') << std::setw(8) << jet_pt << " "
	   << std::setfill(' ') << std::setw(4) << temp2.str() << std::endl;
      }
    }
    if ( edm::isDebugEnabled() ) { LogTrace("TEST") << ss.str(); }
  }
  
  // ---------- Create reco::GenJet and reference pat::Jet collections ----------

  std::vector<GenJet> gen_jets;
  std::vector<pat::Jet> ref_jets;
  {
    uint32_t index = 0;
    std::stringstream ss;
    if ( edm::isDebugEnabled() ) {
      ss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
	 << " ORDERED jets (" << mapping.size() 
	 << ")." << std::endl
	 << " GenJetPt  (#) : RefJetPt  (#)" << std::endl;
    }
    std::map<GenJet,pat::Jet>::const_iterator ii = mapping.begin();
    std::map<GenJet,pat::Jet>::const_iterator jj = mapping.end();
    for ( ; ii != jj; ++ii ) {
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
      if ( edm::isDebugEnabled() ) {
	ss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
	   << " KEPT reco::CaloJets (" << mapping.size() 
	   << ") for \"" << ii->label() << "\"." << std::endl
	   << " RefJetPt  (#) : CaloJetPt  (#)" << std::endl;
      }

      // Retrieve reco::CaloJet collection 
      edm::Handle< edm::View<reco::CaloJet> > jets;
      event.getByLabel( *ii, jets );

      // Sort reco::CaloJet collection according to GenJet pt
      std::auto_ptr< std::vector<reco::CaloJet> > ordered( new std::vector<reco::CaloJet>() );
      std::vector<pat::Jet>::const_iterator iii = ref_jets.begin();
      std::vector<pat::Jet>::const_iterator jjj = ref_jets.end();
      for ( ; iii != jjj; ++iii ) {
	reco::CaloJet temp( iii->p4(), iii->caloSpecific(), iii->getJetConstituents() );
	edm::View<reco::CaloJet>::const_iterator iter = std::find( jets->begin(), jets->end(), temp );
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
      event.put(ordered,ii->label());
      if ( edm::isDebugEnabled() ) { LogTrace("TEST") << ss.str(); }
    }
  }
  
  // ---------- Create reco::PatJet products ----------

  {
    std::vector<edm::InputTag>::const_iterator ii = patJets_.begin();
    std::vector<edm::InputTag>::const_iterator jj = patJets_.end();
    for ( ; ii != jj; ++ii ) {

      // Check for non-null label
      if ( ii->label().empty() ) { continue; }

      // Debug
      std::stringstream ss;
      if ( edm::isDebugEnabled() ) {
	ss << "[jpt::SortJetCollectionsByGenJetPt::" << __func__ << "]"
	   << " KEPT pat::Jets (" << mapping.size() 
	   << ") for \"" << ii->label() << "\"." << std::endl
	   << " RefJetPt  (#) : PatJetPt  (#)" << std::endl;
      }

      // Retrieve pat::Jet collection 
      edm::Handle< edm::View<pat::Jet> > jets;
      event.getByLabel( *ii, jets );

      // Sort pat::Jet collection according to GenJet pt
      std::auto_ptr< std::vector<pat::Jet> > ordered( new std::vector<pat::Jet>() );
      std::vector<pat::Jet>::const_iterator iii = ref_jets.begin();
      std::vector<pat::Jet>::const_iterator jjj = ref_jets.end();
      for ( ; iii != jjj; ++iii ) {
	edm::View<pat::Jet>::const_iterator iter = std::find( jets->begin(), jets->end(), *iii );
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
	     << std::setfill(' ') << std::setw(4) << temp1.str() << " : "
	     << std::setfill(' ') << std::setw(8) << jet_pt << " "
	     << std::setfill(' ') << std::setw(4) << temp2.str() << std::endl;
	}
      }
      event.put(ordered,ii->label());
      if ( edm::isDebugEnabled() ) { LogTrace("TEST") << ss.str(); }
    }
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace jpt;
DEFINE_FWK_MODULE(SortJetCollectionsByGenJetPt);








// // -----------------------------------------------------------------------------
// //
// bool jpt::SortJetCollectionsByGenJetPt::matchJetsByCaloTowers( const pat::Jet& jet1,
// 							  const pat::Jet& jet2 ) {
  
//   std::vector< edm::Ptr<CaloTower> > towers1 = jet1.getCaloConstituents();
//   std::vector< edm::Ptr<CaloTower> > towers2 = jet2.getCaloConstituents();
  
//   if ( towers1.empty() || 
//        towers2.empty() || 
//        towers1.size() != towers2.size() ) { return false; }
  
//   std::vector< edm::Ptr<CaloTower> >::const_iterator ii = towers1.begin();
//   std::vector< edm::Ptr<CaloTower> >::const_iterator jj = towers1.end();
//   for ( ; ii != jj; ++ii ) {
//     std::vector< edm::Ptr<CaloTower> >::const_iterator iii = towers2.begin();
//     std::vector< edm::Ptr<CaloTower> >::const_iterator jjj = towers2.end();
//     for ( ; iii != jjj; ++iii ) { if ( *iii == *ii ) { break; } }
//     if ( iii == towers2.end() ) { return false; }
//   }
  
//   return true;
  
// }
