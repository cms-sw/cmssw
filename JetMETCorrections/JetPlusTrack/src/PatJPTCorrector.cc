#include "JetMETCorrections/JetPlusTrack/interface/PatJPTCorrector.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
//#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRCalo.h"
//#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertex.h"
//#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include <fstream>
#include <sstream>
#include <vector>

//@@ JTA on-the-fly
//   <use name=MagneticField/Records>
//   <use name=RecoJets/JetAssociationAlgorithms>
//   <use name=TrackingTools/Records>

using namespace std;
using namespace jpt;

// -----------------------------------------------------------------------------
//
PatJPTCorrector::PatJPTCorrector( const edm::ParameterSet& pset ) 
  : JetPlusTrackCorrector(pset),
    usePat_( pset.getParameter<bool>("UsePatCollections") ),
    allowOnTheFly_( pset.getParameter<bool>("AllowOnTheFly") ),
    tracks_( pset.getParameter<edm::InputTag>("Tracks") ),
    propagator_( pset.getParameter<std::string>("Propagator") ),
    coneSize_( pset.getParameter<double>("ConeSize") )
{;}

// -----------------------------------------------------------------------------
//
PatJPTCorrector::~PatJPTCorrector() {;}

// -----------------------------------------------------------------------------
//
bool PatJPTCorrector::jetTrackAssociation( const reco::Jet& fJet,
					   const edm::Event& event, 
					   const edm::EventSetup& setup,
					   JetTracks& trks ) const {
  
  // Some init
  trks.clear();
  
  // Check if labels are given
  if ( !jetTracksAtVertex_.label().empty() && !jetTracksAtCalo_.label().empty() ) { 
    return jtaUsingEventData( fJet, event, trks );
  } else {
    return jtaOnTheFly( fJet, event, setup, trks );
  }
  
  return false;
  
}

// -----------------------------------------------------------------------------
//
bool PatJPTCorrector::jtaOnTheFly( const reco::Jet& fJet,
				   const edm::Event& event, 
				   const edm::EventSetup& setup,
				   JetTracks& trks ) const {

  edm::LogWarning("PatJPTCorrector") 
    << "[PatJPTCorrector::" << __func__ << "]"
    << " Please provide valid InputTags for the reco::JetTracksAssociation::Containers!"
    << " (\"On-the-fly\" mode not yet implemented.)";
  return false;
  
//   if ( !allowOnTheFly_ ) {

//     edm::LogWarning("PatJPTCorrector") 
//       << "[PatJPTCorrector::" << __func__ << "]"
//       << " \"On-the-fly\" mode not allowed by configuration!...";

//     return false;

//   } else {
    
//     // Construct objects that perform association 
//     static JetTracksAssociationDRVertex vrtx(coneSize_);
//     static JetTracksAssociationDRCalo   calo(coneSize_);
  
//     // Container for propagated tracks
//     static JetTracksAssociationDR::TrackRefs propagated;
    
//     // Perform once per event
//     static uint32_t last_event = 0;
//     if ( event.id().event() != last_event ) {
//       last_event = event.id().event();

//       // Retrieve magnetic field and track propagator
//       edm::ESHandle<MagneticField> field;
//       setup.get<IdealMagneticFieldRecord>().get( field );
//       edm::ESHandle<Propagator> propagator;
//       setup.get<TrackingComponentsRecord>().get( propagator_, propagator );

//       // Retrieve global tracks 
//       edm::Handle<reco::TrackCollection> tracks;
//       event.getByLabel( tracks_, tracks );
//       if ( !tracks.isValid() || tracks.failedToGet() ) {
//         edm::LogError("PatJPTCorrector")
// 	  << "[PatJPTCorrector::" << __func__ << "]"
// 	  << " Invalid handle to \"reco::TrackCollection\""
// 	  << " with InputTag (label:instance:process) \"" 
// 	  << tracks_.label() << ":"
// 	  << tracks_.instance() << ":"
// 	  << tracks_.process() << "\"";
//         return false;
//       }

//       // Propagate tracks for to calo face 
//       JetTracksAssociationDR::createTrackRefs( propagated, tracks, trackQuality_ );
//       vrtx.propagateTracks( propagated ); //@@ needed?
//       calo.propagateTracks( propagated, *field, *propagator );
      
//     } 

//     // Associate tracks to jets at both vertex and calo face
//     vrtx.associateTracksToJet( trks.vertex_, fJet, propagated );
//     calo.associateTracksToJet( trks.caloFace_, fJet, propagated );
    
//     // Check if any tracks are associated to jet at vertex
//     if ( trks.vertex_.empty() ) { return false; }
    
//     return true;

//   }

}

// -----------------------------------------------------------------------------
//
void PatJPTCorrector::matchTracks( const JetTracks& jet_tracks, 
				   const edm::Event& event, 
				   MatchedTracks& pions, 
				   MatchedTracks& muons,
				   MatchedTracks& elecs ) const { 
  
  // Some init  
  pions.clear(); 
  muons.clear(); 
  elecs.clear(); 

  // Get PAT muons
  edm::Handle<PatMuons> pat_muons;
  bool found_pat_muons = true;
  if ( useMuons_ ) { getMuons( event, pat_muons ); }
  
  // Get PAT electrons and their ids
  edm::Handle<PatElectrons> pat_elecs;
  bool found_pat_elecs = true;
  if ( useElecs_ ) { getElectrons( event, pat_elecs ); }

  // Check PAT products found
  if ( !found_pat_muons || !found_pat_elecs ) {
    edm::LogError("PatJPTCorrector")
      << "[PatJPTCorrector::" << __func__ << "]"
      << " Unable to access PAT collections for muons and electrons";
    return;
  }
  
  // Identify pions/muons/electrons that are "in/in" and "in/out"
  {
    TrackRefs::const_iterator itrk = jet_tracks.vertex_.begin();
    TrackRefs::const_iterator jtrk = jet_tracks.vertex_.end();
    for ( ; itrk != jtrk; ++itrk ) {

      if ( failTrackQuality(itrk) ) { continue; }
      
      TrackRefs::iterator it = jet_tracks.caloFace_.end();
      bool found = findTrack( jet_tracks, itrk, it );

      bool is_muon = useMuons_ && matchMuons( itrk, pat_muons );
      bool is_ele  = useElecs_ && matchElectrons( itrk, pat_elecs );

      if ( found ) { 
	if ( is_muon )     { muons.inVertexInCalo_.push_back(*it); }
	else if ( is_ele ) { elecs.inVertexInCalo_.push_back(*it); } 
	else               { pions.inVertexInCalo_.push_back(*it); } 
      } else { 
	if ( is_muon )     { muons.inVertexOutOfCalo_.push_back(*itrk); }
	//else if ( is_ele ) { elecs.inVertexOutOfCalo_.push_back(*itrk); } //@@ bug?  
	else               { pions.inVertexOutOfCalo_.push_back(*itrk); }
      } 
    } 
  }
  
  // Identify pions/muons/electrons that are "out/in"
  {
    TrackRefs::iterator itrk = jet_tracks.caloFace_.begin(); 
    TrackRefs::iterator jtrk = jet_tracks.caloFace_.end(); 
    for ( ; itrk != jtrk; ++itrk ) {
      
      if ( failTrackQuality(itrk) ) { continue; }
      
      if ( !tracksInCalo( pions, muons, elecs ) ) { continue; }

      bool found = findTrack( pions, muons, elecs, itrk );
      
      if ( !found ) {
	
	bool is_muon = useMuons_ && matchMuons( itrk, pat_muons );
	bool is_ele  = false; //@@ bug? useElecs_ && matchElectrons( itrk, pat_elecs );
	
	if ( is_muon )     { muons.outOfVertexInCalo_.push_back(*itrk); } 
	else if ( is_ele ) { elecs.outOfVertexInCalo_.push_back(*itrk); } //@@ bug?
	else               { pions.outOfVertexInCalo_.push_back(*itrk); }
	
      }
    } 
  }
  
  if ( verbose_ && edm::isDebugEnabled() ) {
    std::stringstream ss;
    ss << "[PatJPTCorrector::" << __func__ << "] Number of tracks:" << std::endl 
       << " In-cone at Vertex and in-cone at CaloFace:" << std::endl  
       << "  Pions      : " << pions.inVertexInCalo_.size() << std::endl
       << "  Muons      : " << muons.inVertexInCalo_.size() << std::endl
       << "  Electrons  : " << elecs.inVertexInCalo_.size() << std::endl
       << " In-cone at Vertex and out-of-cone at CaloFace:" << std::endl  
       << "  Pions      : " << pions.inVertexOutOfCalo_.size() << std::endl
       << "  Muons      : " << muons.inVertexOutOfCalo_.size() << std::endl
       << "  Electrons  : " << elecs.inVertexOutOfCalo_.size() << std::endl
       << " Out-of-cone at Vertex and in-cone at CaloFace:" << std::endl  
       << "  Pions      : " << pions.outOfVertexInCalo_.size() << std::endl
       << "  Muons      : " << muons.outOfVertexInCalo_.size() << std::endl
       << "  Electrons  : " << elecs.outOfVertexInCalo_.size() << std::endl;
    LogTrace("PatJPTCorrector") << ss.str();
  }
  
}

// -----------------------------------------------------------------------------
//
bool PatJPTCorrector::getMuons( const edm::Event& event, edm::Handle<PatMuons>& pat_muons ) const {
  event.getByLabel( muons_, pat_muons ); 
  if ( !pat_muons.isValid() || pat_muons.failedToGet() ) {
    edm::LogError("PatJPTCorrector")
      << "[PatJPTCorrector::" << __func__ << "]"
      << " Invalid handle to pat::Muon collection"
      << " with InputTag (label:instance:process) \"" 
      << muons_.label() << ":"
      << muons_.instance() << ":"
      << muons_.process() << "\"";
    return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
//
bool PatJPTCorrector::getElectrons( const edm::Event& event, edm::Handle<PatElectrons>& pat_elecs ) const {
  event.getByLabel( electrons_, pat_elecs ); 
  if ( !pat_elecs.isValid() || pat_elecs.failedToGet() ) {
    edm::LogError("PatJPTCorrector")
      << "[PatJPTCorrector::" << __func__ << "]"
      << " Invalid handle to pat::Electron collection"
      << " with InputTag (label:instance:process) \"" 
      << electrons_.label() << ":"
      << electrons_.instance() << ":"
      << electrons_.process() << "\"";
    return false;
  }
  return true;
} 

// -----------------------------------------------------------------------------
//
bool PatJPTCorrector::matchMuons( reco::TrackRefVector::const_iterator itrk, 
				  const edm::Handle<PatMuons>& muons ) const {
  
  if ( muons->empty() ) { return false; }
  
  PatMuons::const_iterator imuon = muons->begin(); 
  PatMuons::const_iterator jmuon = muons->end(); 
  for ( ; imuon != jmuon; ++imuon ) {

    if ( imuon->innerTrack().isNull() ||
	 !muon::isGoodMuon(*imuon,muon::TMLastStationTight) ||
	 imuon->innerTrack()->pt() < 3.0 ) { continue; }
    
    if ( itrk->id() != imuon->innerTrack().id() ) {
      edm::LogError("PatJPTCorrector")
	<< "Product id of the tracks associated to the jet " << itrk->id() 
	<<" is different from the product id of the inner track used for muons " << imuon->innerTrack().id()
	<< "!" << std::endl
	<< "Cannot compare tracks from different collection. Configuration Error!";
      return false;
    }
    
    if ( *itrk == imuon->innerTrack() ) { return true; }
  }
  
  return false;
  
}

// -----------------------------------------------------------------------------
//
bool PatJPTCorrector::matchElectrons( reco::TrackRefVector::const_iterator itrk, 
				      const edm::Handle<PatElectrons>& electrons ) const {

  if ( electrons->empty() ) { return false; }
  
  double deltaR = 999.;
  double deltaRMIN = 999.;
  
  PatElectrons::const_iterator ielec = electrons->begin(); 
  PatElectrons::const_iterator jelec = electrons->end(); 
  for ( ; ielec != jelec; ++ielec ) {
    
    if ( ielec->electronID( electronIds_.label() ) < 1.e-6 ) { continue; } //@@ Check for null value 
    
    // DR matching b/w electron and track
    double deltaphi = fabs( ielec->phi() - (*itrk)->momentum().phi() );
    if ( deltaphi > 6.283185308 ) deltaphi -= 6.283185308;
    if ( deltaphi > 3.141592654 ) deltaphi = 6.283185308 - deltaphi;
    deltaR = abs( sqrt( pow( (ielec->eta() - (*itrk)->momentum().eta()), 2 ) + 
			pow( deltaphi , 2 ) ) ); 
    if ( deltaR < deltaRMIN ) { deltaRMIN = deltaR; }
    
  }
  
  if ( deltaR < 0.02 ) return true;
  else return false;
  
}










  
//   // Some init  
//   pions.clear(); 
//   muons.clear(); 
//   electrons.clear(); 

//   // Get muons
//   bool found_reco_muons = true;
//   bool found_pat_muons  = true;
//   edm::Handle<RecoMuons> reco_muons;
//   edm::Handle<PatMuons> pat_muons;
//   if ( useMuons_ ) { 

//     if ( !usePat_ ) {

//       // Get RECO muons
//       event.getByLabel( muons_, reco_muons ); 
//       if ( !reco_muons.isValid() || reco_muons.failedToGet() ) {
// 	found_reco_muons = false;
// 	edm::LogError("PatJPTCorrector")
// 	  << "[PatJPTCorrector::" << __func__ << "]"
// 	  << " Invalid handle to reco::GsfMuon collection"
// 	  << " with InputTag (label:instance:process) \"" 
// 	  << muons_.label() << ":"
// 	  << muons_.instance() << ":"
// 	  << muons_.process() << "\"";
//       }

//     } else { 
      
//       // Get PAT muons
//       event.getByLabel( muons_, pat_muons ); 
//       if ( !pat_muons.isValid() || pat_muons.failedToGet() ) {
// 	found_pat_muons = false;
// 	edm::LogError("PatJPTCorrector")
// 	  << "[PatJPTCorrector::" << __func__ << "]"
// 	  << " Invalid handle to pat::Muon collection"
// 	  << " with InputTag (label:instance:process) \"" 
// 	  << muons_.label() << ":"
// 	  << muons_.instance() << ":"
// 	  << muons_.process() << "\"";
//       } 

//     }

//   } 
  
//   // Get electrons
//   bool found_reco_electrons    = true;
//   bool found_reco_electron_ids = true;
//   bool found_pat_electrons     = true;
//   edm::Handle<RecoElectrons> reco_electrons;
//   edm::Handle<RecoElectronIDs> reco_electron_ids;
//   edm::Handle<PatElectrons> pat_electrons;
//   if ( useElectrons_ ) { 

//     if ( !usePat_ ) {

//       // Get RECO electrons
//       event.getByLabel( electrons_, reco_electrons ); 
//       if ( !reco_electrons.isValid() || reco_electrons.failedToGet() ) {
// 	found_reco_electrons = false;
// 	edm::LogError("PatJPTCorrector")
// 	  << "[PatJPTCorrector::" << __func__ << "]"
// 	  << " Invalid handle to reco::GsfElectron collection"
// 	  << " with InputTag (label:instance:process) \"" 
// 	  << electrons_.label() << ":"
// 	  << electrons_.instance() << ":"
// 	  << electrons_.process() << "\"";
//       }

//       // Get RECO electron IDs
//       event.getByLabel( electronIds_, reco_electron_ids ); 
//       if ( !reco_electron_ids.isValid() || reco_electron_ids.failedToGet() ) {
// 	found_reco_electron_ids = false;
// 	edm::LogError("PatJPTCorrector")
// 	  << "[PatJPTCorrector::" << __func__ << "]"
// 	  << " Invalid handle to reco::GsfElectron collection"
// 	  << " with InputTag (label:instance:process) \"" 
// 	  << electronIds_.label() << ":"
// 	  << electronIds_.instance() << ":"
// 	  << electronIds_.process() << "\"";
//       }

//     } else { 
      
//       // Get PAT electrons
//       event.getByLabel( electrons_, pat_electrons ); 
//       if ( !pat_electrons.isValid() || pat_electrons.failedToGet() ) {
// 	found_pat_electrons = false;
// 	edm::LogError("PatJPTCorrector")
// 	  << "[PatJPTCorrector::" << __func__ << "]"
// 	  << " Invalid handle to pat::Electron collection"
// 	  << " with InputTag (label:instance:process) \"" 
// 	  << electrons_.label() << ":"
// 	  << electrons_.instance() << ":"
// 	  << electrons_.process() << "\"";
//       }

//     }

//   } 
  
//   // Check
//   bool found_reco = found_reco_muons && found_reco_electrons && found_reco_electron_ids;
//   bool found_pat  = found_pat_muons && found_pat_electrons;
//   if ( !found_reco && !found_pat ) {
//     edm::LogError("PatJPTCorrector")
//       << "[PatJPTCorrector::" << __func__ << "]"
//       << " Unable to access RECO or PAT collections for muons and electrons";
//     return;
//   }
  
//   // Loop through tracks at "Vertex"
//   {
//     reco::TrackRefVector::const_iterator itrk = jet_tracks.vertex_.begin();
//     reco::TrackRefVector::const_iterator jtrk = jet_tracks.vertex_.end();
//     for ( ; itrk != jtrk; ++itrk ) {

//       if ( useTrackQuality_ && !(*itrk)->quality(trackQuality_) ) { continue; }
      
//       reco::TrackRefVector::iterator it = find( jet_tracks.caloFace_.begin(),
// 						jet_tracks.caloFace_.end(),
// 						*itrk );
      
//       bool is_muon = false;
//       bool is_ele  = false;
//       if ( !usePat_ && found_reco ) { 
// 	is_muon = useMuons_     && matching( itrk, reco_muons );
// 	is_ele  = useElectrons_ && matching( itrk, reco_electrons, reco_electron_ids );
//       } else if ( usePat_ && found_pat ) { 
// 	is_muon = useMuons_     && matching( itrk, pat_muons ); 
// 	is_ele  = useElectrons_ && matching( itrk, pat_electrons );
//       }

//       //@@ bug? 
//       if ( it != jet_tracks.caloFace_.end() ) { 
// 	if ( is_muon )     { muons.inVertexInCalo_.push_back(*it); }
// 	else if ( is_ele ) { electrons.inVertexInCalo_.push_back(*it); } 
// 	else               { pions.inVertexInCalo_.push_back(*it); } 
//       } else { 
// 	if ( is_muon )     { muons.inVertexOutOfCalo_.push_back(*itrk); }
// 	//else if ( is_ele ) { electrons.inVertexOutOfCalo_.push_back(*itrk); } 
// 	else               { pions.inVertexOutOfCalo_.push_back(*itrk); }
//       } 
      
//     } 
    
//   }
  
//   // Loop through tracks at "CaloFace"
//   {
//     reco::TrackRefVector::iterator itrk = jet_tracks.caloFace_.begin(); 
//     reco::TrackRefVector::iterator jtrk = jet_tracks.caloFace_.end(); 

//     for ( ; itrk != jtrk; ++itrk ) {
      
//       if ( useTrackQuality_ && !(*itrk)->quality(trackQuality_) ) { continue; }
      
//       //@@ bug?
//       if( !pions.inVertexInCalo_.empty() ) { //||
// 	//!muons.inVertexInCalo_.empty() || 
// 	//!electrons.inVertexInCalo_.empty() ) { 
	
// 	reco::TrackRefVector::iterator it = find( pions.inVertexInCalo_.begin(),
// 						  pions.inVertexInCalo_.end(),
// 						  *itrk );
	
// 	reco::TrackRefVector::iterator im = find( muons.inVertexInCalo_.begin(),
// 						  muons.inVertexInCalo_.end(),
// 						  *itrk );
	
// 	//@@ bug?
// 	// 	reco::TrackRefVector::iterator ie = find( electrons.inVertexInCalo_.begin(),
// 	// 						  electrons.inVertexInCalo_.end(),
// 	// 						  *itrk );

// 	//@@ bug?	
// 	if ( it == pions.inVertexInCalo_.end() && 
// 	     im == muons.inVertexInCalo_.end() ) { //&&
// 	  //ie == electrons.inVertexInCalo_.end() ) {

// 	  //@@ bug?
// 	  bool is_muon = false;
// 	  bool is_ele  = false;
// 	  if ( !usePat_ && found_reco ) { 
// 	    is_muon = useMuons_ && matching( itrk, reco_muons );
// 	    is_ele  = false; //useElectrons_ && matching( itrk, reco_electrons, reco_electron_ids );
// 	  } else if ( usePat_ && found_pat ) { 
// 	    is_muon = useMuons_ && matching( itrk, pat_muons ); 
// 	    is_ele  = false; //useElectrons_ && matching( itrk, pat_electrons );
// 	  }
	  
// 	  if ( is_muon )     { muons.outOfVertexInCalo_.push_back(*itrk); } 
// 	  else if ( is_ele ) { electrons.outOfVertexInCalo_.push_back(*itrk); } //@@ bug?
// 	  else               { pions.outOfVertexInCalo_.push_back(*itrk); }
	  
// 	}
//       }
//     } 
    
//   }
  
//   if ( verbose_ && edm::isDebugEnabled() ) {
//     std::stringstream ss;
//     ss << "[PatJPTCorrector::" << __func__ << "] Number of tracks:" << std::endl 
//        << " In-cone at Vertex and in-cone at CaloFace:" << std::endl  
//        << "  Pions      : " << pions.inVertexInCalo_.size() << std::endl
//        << "  Muons      : " << muons.inVertexInCalo_.size() << std::endl
//        << "  Electrons  : " << electrons.inVertexInCalo_.size() << std::endl
//        << " In-cone at Vertex and out-of-cone at CaloFace:" << std::endl  
//        << "  Pions      : " << pions.inVertexOutOfCalo_.size() << std::endl
//        << "  Muons      : " << muons.inVertexOutOfCalo_.size() << std::endl
//        << "  Electrons  : " << electrons.inVertexOutOfCalo_.size() << std::endl
//        << " Out-of-cone at Vertex and in-cone at CaloFace:" << std::endl  
//        << "  Pions      : " << pions.outOfVertexInCalo_.size() << std::endl
//        << "  Muons      : " << muons.outOfVertexInCalo_.size() << std::endl
//        << "  Electrons  : " << electrons.outOfVertexInCalo_.size() << std::endl;
//     LogTrace("PatJPTCorrector") << ss.str();
//   }
