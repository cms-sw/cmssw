#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
//#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRCalo.h"
//#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertex.h"
//#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
using namespace jpt;

// -----------------------------------------------------------------------------
//
JetPlusTrackCorrector::JetPlusTrackCorrector( const edm::ParameterSet& pset ) 
  : verbose_( pset.getParameter<bool>("Verbose") ),
    useInConeTracks_( pset.getParameter<bool>("UseInConeTracks") ),
    useOutOfConeTracks_( pset.getParameter<bool>("UseOutOfConeTracks") ),
    useOutOfVertexTracks_( pset.getParameter<bool>("UseOutOfVertexTracks") ),
    useMuons_( pset.getParameter<bool>("UseMuons") ),
    useElectrons_( pset.getParameter<bool>("UseElectrons") ),
    usePat_( pset.getParameter<bool>("UsePatCollections") ),
    allowOnTheFly_( pset.getParameter<bool>("AllowOnTheFly") ),
    useTrackQuality_( pset.getParameter<bool>("UseTrackQuality") ),
    jetTracksAtVertex_( pset.getParameter<edm::InputTag>("JetTracksAssociationAtVertex") ),
    jetTracksAtCalo_( pset.getParameter<edm::InputTag>("JetTracksAssociationAtCaloFace") ),
    jetSplitMerge_( pset.getParameter<int>("JetSplitMerge") ),
    tracks_( pset.getParameter<edm::InputTag>("Tracks") ),
    propagator_( pset.getParameter<std::string>("Propagator") ),
    coneSize_( pset.getParameter<double>("ConeSize") ),
    muons_( pset.getParameter<edm::InputTag>("Muons") ),
    electrons_( pset.getParameter<edm::InputTag>("Electrons") ),
    electronIds_( pset.getParameter<edm::InputTag>("ElectronIds") ),
    trackQuality_( reco::TrackBase::qualityByName( pset.getParameter<std::string>("TrackQuality") ) ),
    response_( new Map( pset.getParameter<std::string>("ResponseMap"), verbose_ ) ),
    efficiency_( new Map( pset.getParameter<std::string>("EfficiencyMap"), verbose_ ) ),
    leakage_( new Map( pset.getParameter<std::string>("LeakageMap"), verbose_ ) )
{

  if ( !response_ ) {
    edm::LogError("JetPlusTrackCorrector")
      << "[JetPlusTrackCorrector::" << __func__ << "]"
      << " NULL pointer to response map!";
  }

  if ( !efficiency_ ) {
    edm::LogError("JetPlusTrackCorrector")
      << "[JetPlusTrackCorrector::" << __func__ << "]"
      << " NULL pointer to efficiency map!";
  }

  if ( !leakage_ ) {
    edm::LogError("JetPlusTrackCorrector")
      << "[JetPlusTrackCorrector::" << __func__ << "]"
      << " NULL pointer to leakage map!";
  }
  
  if ( !useInConeTracks_ || 
       !useOutOfConeTracks_ ||
       !useOutOfVertexTracks_ ) {
    std::stringstream ss;
    ss << "[JetPlusTrackCorrector::" << __func__ << "]"
       << " You are using JPT algorithm in a non-standard way!" << std::endl
       << " UseInConeTracks      : " << ( useInConeTracks_ ? "true" : "false" ) << std::endl
       << " UseOutOfConeTracks   : " << ( useOutOfConeTracks_ ? "true" : "false" ) << std::endl
       << " UseOutOfVertexTracks : " << ( useOutOfVertexTracks_ ? "true" : "false" );
    edm::LogWarning("JetPlusTrackCorrector") << ss.str();
  }

}

// -----------------------------------------------------------------------------
//
JetPlusTrackCorrector::~JetPlusTrackCorrector() {
  if ( response_ ) { delete response_; } 
  if ( efficiency_ ) { delete efficiency_; }
  if ( leakage_ ) { delete leakage_; }
}

// -----------------------------------------------------------------------------
//
double JetPlusTrackCorrector::correction( const reco::Jet& fJet,
				 const edm::Event& event,
				 const edm::EventSetup& setup ) const 
{
  
  // Jet energy to correct
  double jet_energy = fJet.energy();
  
  // Check that jet falls within |eta| < 2.1
  if ( fabs( fJet.eta() ) > 2.1 ) { return jet_energy / fJet.energy(); }
  
  // Associate tracks to jet at both the Vertex and CaloFace
  AssociatedTracks associated_tracks;
  bool ok = jetTrackAssociation( fJet, event, setup, associated_tracks ); 
  if ( !ok ) { return ( jet_energy / fJet.energy() ); }

  // Track collections propagated to Vertex and CaloFace for "pions", muons and electrons
  ParticleTracks pions;
  ParticleTracks muons;
  ParticleTracks electrons;
  particles( associated_tracks, event, setup, pions, muons, electrons );

  // -------------------- Pions --------------------
  
  ParticleResponse in_cone;
  ParticleResponse out_of_cone;
  ParticleResponse out_of_vertex;
  double corr_pions_in_cone = 0.;
  double corr_pions_out_of_cone = 0.;
  double corr_pions_out_of_vertex = 0.;
  
  if ( useInConeTracks_ ) { 
    corr_pions_in_cone = correction( pions.inVertexInCalo_, in_cone, true, true ); 
    jet_energy += corr_pions_in_cone;
  }
  
  if ( useOutOfConeTracks_ ) {
    corr_pions_out_of_cone = correction( pions.inVertexOutOfCalo_, out_of_cone, true, false );
    jet_energy += corr_pions_out_of_cone;
  }

  if ( useOutOfVertexTracks_ ) {
    corr_pions_out_of_vertex = correction( pions.outOfVertexInCalo_, out_of_vertex, false, true );
    jet_energy += corr_pions_out_of_vertex;
  }

  // -------------------- Efficiency --------------------
  
  double corr_pion_eff_in_cone = 0.;
  double corr_pion_eff_out_of_cone = 0.;
  
  if ( useInConeTracks_ ) { 
    corr_pion_eff_in_cone = correction( in_cone, true );
    jet_energy += corr_pion_eff_in_cone;
  }

  if ( useOutOfConeTracks_ ) {
    corr_pion_eff_out_of_cone = correction( out_of_cone, false );
    jet_energy += corr_pion_eff_out_of_cone;
  }
  
  // -------------------- Muons --------------------
  
  ParticleResponse not_used1;
  ParticleResponse not_used2;
  ParticleResponse not_used3;
  double corr_muons_in_cone = 0.;
  double corr_muons_out_of_cone = 0.;
  double corr_muons_out_of_vertex = 0.;
  
  if ( useInConeTracks_ ) { 
    corr_muons_in_cone = correction( muons.inVertexInCalo_, not_used1, true, true, 0.105, 2. );
    jet_energy += corr_muons_in_cone;
  }  
  
  if ( useOutOfConeTracks_ ) {
    if ( !pions.inVertexOutOfCalo_.empty() ) { //@@ bug?
      corr_muons_out_of_cone = correction( muons.inVertexOutOfCalo_, not_used2, true, false, 0.105, 2. );
      jet_energy += corr_muons_out_of_cone;
    }
  }    
  
  if ( useOutOfVertexTracks_ ) {
    corr_muons_out_of_vertex = correction( muons.outOfVertexInCalo_, not_used3, false, true, 0.105, 2. );
    jet_energy += corr_muons_out_of_vertex;
  }
  
  // -------------------- Return corrected energy -------------------- 

  if ( verbose_ ) {
    std::stringstream ss;
    ss << "[JetPlusTrackCorrector::" << __func__ << "] Corrections summary:" << std::endl 
       << "Number of tracks:" << std::endl
       << " In-cone at Vertex   : " << associated_tracks.atVertex_.size() << std::endl
       << " In-cone at CaloFace : " << associated_tracks.atCaloFace_.size() << std::endl
       << "Individual corrections (and number of tracks):" << std::endl
       << " In-cone at Vertex and in-cone at CaloFace (subtract response, add track momentum):" << std::endl  
       << "  Pions      : " << "(" << pions.inVertexInCalo_.size() << ") " << corr_pions_in_cone << std::endl  
       << "  Muons      : " << "(" << muons.inVertexInCalo_.size() << ") " << corr_muons_in_cone << std::endl  
       << "  Electrons  : " << "(" << electrons.inVertexInCalo_.size() << ") " << double(0.) << std::endl  
       << "  Efficiency : " << "(" << pions.inVertexInCalo_.size() << ") " << corr_pion_eff_in_cone << std::endl  
       << " In-cone at Vertex and out-of-cone at CaloFace (add track momentum):" << std::endl  
       << "  Pions      : " << "(" << pions.inVertexOutOfCalo_.size() << ") " << corr_pions_out_of_cone << std::endl  
       << "  Muons      : " << "(" << muons.inVertexOutOfCalo_.size() << ") " << corr_muons_out_of_cone << std::endl  
       << "  Electrons  : " << "(" << electrons.inVertexOutOfCalo_.size() << ") " << double(0.) << std::endl  
       << "  Efficiency : " << "(" << pions.inVertexOutOfCalo_.size() << ") " << corr_pion_eff_out_of_cone << std::endl  
       << " Out-of-cone at Vertex and in-cone at CaloFace (subtract response):" << std::endl  
       << "  Pions      : " << "(" << pions.outOfVertexInCalo_.size() << ") " << corr_pions_out_of_vertex << std::endl  
       << "  Muons      : " << "(" << muons.outOfVertexInCalo_.size() << ") " << corr_muons_out_of_vertex << std::endl 
       << "  Electrons  : " << "(" << electrons.outOfVertexInCalo_.size() << ") " << double(0.) << std::endl  
       << "Total correction:"
       << " Uncorrected energy : " << fJet.energy() << std::endl
       << " Correction factor  : " << jet_energy / fJet.energy() << std::endl
       << " Corrected energy   : " << jet_energy;
    edm::LogVerbatim("JetPlusTrackCorrector") << ss.str();
  }
  
  // Check if scale is negative
  double scale = jet_energy / fJet.energy();
  if ( scale < 0. ) { scale = 1.; } 

//   LogTrace("test") << " mScale= " << scale
//    		   << " NewResponse " << jet_energy 
//    		   << " Jet energy " << fJet.energy()
//    		   << " event " << event.id().event();
  
  return scale;
  
}

// -----------------------------------------------------------------------------
//
double JetPlusTrackCorrector::correction( const reco::Jet& jet ) const {
  edm::LogError("JetPlusTrackCorrector")
    << "JetPlusTrackCorrector can be run on entire event only";
  return 1.;
}

// -----------------------------------------------------------------------------
//
double JetPlusTrackCorrector::correction( const reco::Particle::LorentzVector& jet ) const {
  edm::LogError("JetPlusTrackCorrector")
    << "JetPlusTrackCorrector can be run on entire event only";
  return 1.;
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::jetTrackAssociation( const reco::Jet& fJet,
					const edm::Event& event, 
					const edm::EventSetup& setup,
					AssociatedTracks& trks ) const {
  
  // Some init
  trks.clear();
  
  // Check whether to retrieve JTA object from Event or construct "on-the-fly"
  if ( !jetTracksAtVertex_.label().empty() && 
       !jetTracksAtCalo_.label().empty() ) { 
    
    // Get Jet-track association at Vertex
    edm::Handle<reco::JetTracksAssociation::Container> jetTracksAtVertex;
    event.getByLabel( jetTracksAtVertex_, jetTracksAtVertex ); 
    if ( !jetTracksAtVertex.isValid() || jetTracksAtVertex.failedToGet() ) {
      if ( verbose_ && edm::isDebugEnabled() ) {
	edm::LogWarning("JetPlusTrackCorrector")
	  << "[JetPlusTrackCorrector::" << __func__ << "]"
	  << " Invalid handle to reco::JetTracksAssociation::Container (for Vertex)"
	  << " with InputTag (label:instance:process) \"" 
	  << jetTracksAtVertex_.label() << ":"
	  << jetTracksAtVertex_.instance() << ":"
	  << jetTracksAtVertex_.process() << "\"" << std::endl
	  << " Attempting to use JTA \"on-the-fly\" mode...";
      }
      return jtaOnTheFly( fJet, event, setup, trks );
    }
    
    // Retrieve jet-tracks association for given jet
    const reco::JetTracksAssociation::Container jtV = *( jetTracksAtVertex.product() );
    reco::TrackRefVector excluded; 
    if ( jetSplitMerge_ < 0 ) { trks.atVertex_ = reco::JetTracksAssociation::getValue( jtV, fJet ); }
    else { rebuildJta( fJet, jtV, trks.atVertex_, excluded ); }
    
    // Check if any tracks are associated to jet at vertex
    if ( trks.atVertex_.empty() ) { return false; }

    // Get Jet-track association at Calo
    edm::Handle<reco::JetTracksAssociation::Container> jetTracksAtCalo;
    event.getByLabel( jetTracksAtCalo_, jetTracksAtCalo ); 
    if ( !jetTracksAtCalo.isValid() || jetTracksAtCalo.failedToGet() ) {
      if ( verbose_ && edm::isDebugEnabled() ) {
	edm::LogWarning("JetPlusTrackCorrector")
	  << "[JetPlusTrackCorrector::" << __func__ << "]"
	  << " Invalid handle to reco::JetTracksAssociation::Container (for CaloFace)"
	  << " with InputTag (label:instance:process) \"" 
	  << jetTracksAtCalo_.label() << ":"
	  << jetTracksAtCalo_.instance() << ":"
	  << jetTracksAtCalo_.process() << "\"" << std::endl
	  << " Attempting to use JTA \"on-the-fly\" mode...";
      }
      return jtaOnTheFly( fJet, event, setup, trks );
    }
    
    // Retrieve jet-tracks association for given jet
    const reco::JetTracksAssociation::Container jtC = *( jetTracksAtCalo.product() );
    if ( jetSplitMerge_ < 0 ) { trks.atCaloFace_ = reco::JetTracksAssociation::getValue( jtC, fJet ); }
    else { excludeJta( fJet, jtC, trks.atCaloFace_, excluded ); }
    
    // Successful
    return true;
    
  } else { return jtaOnTheFly( fJet, event, setup, trks ); }
  
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::jtaOnTheFly( const reco::Jet& fJet,
				const edm::Event& event, 
				const edm::EventSetup& setup,
				AssociatedTracks& trks ) const {

  edm::LogWarning("JetPlusTrackCorrector") 
    << "[JetPlusTrackCorrector::" << __func__ << "]"
    << " \"On-the-fly\" mode not yet implemented!...";
  return false;
  
//   // Construct objects that perform association 
//   static JetTracksAssociationDRVertex vrtx(coneSize_);
//   static JetTracksAssociationDRCalo   calo(coneSize_);
  
//   // Container for propagated tracks
//   static JetTracksAssociationDR::TrackRefs propagated;
    
//   // Perform once per event
//   static uint32_t last_event = 0;
//   if ( event.id().event() != last_event ) {
//     last_event = event.id().event();

//     // Retrieve magnetic field and track propagator
//     edm::ESHandle<MagneticField> field;
//     setup.get<IdealMagneticFieldRecord>().get( field );
//     edm::ESHandle<Propagator> propagator;
//     setup.get<TrackingComponentsRecord>().get( propagator_, propagator );

//     // Retrieve global tracks 
//     edm::Handle<reco::TrackCollection> tracks;
//     event.getByLabel( tracks_, tracks );
//     if ( !tracks.isValid() || tracks.failedToGet() ) {
//       edm::LogError("JetPlusTrackCorrector")
// 	<< "[JetPlusTrackCorrector::" << __func__ << "]"
// 	<< " Invalid handle to \"reco::TrackCollection\""
// 	<< " with InputTag (label:instance:process) \"" 
// 	<< tracks_.label() << ":"
// 	<< tracks_.instance() << ":"
// 	<< tracks_.process() << "\"";
//       return false;
//     }

//     // Propagate tracks for to calo face 
//     JetTracksAssociationDR::createTrackRefs( propagated, tracks, trackQuality_ );
//     vrtx.propagateTracks( propagated ); //@@ needed?
//     calo.propagateTracks( propagated, *field, *propagator );
      
//   } 

//   // Associate tracks to jets at both vertex and calo face
//   vrtx.associateTracksToJet( trks.atVertex_, fJet, propagated );
//   calo.associateTracksToJet( trks.atCaloFace_, fJet, propagated );
    
//   // Check if any tracks are associated to jet at vertex
//   if ( trks.atVertex_.empty() ) { return false; }
    
//   return true;

}

// -----------------------------------------------------------------------------
//
void JetPlusTrackCorrector::rebuildJta( const reco::Jet& fJet, 
			       const JetTracksAssociations& jtV0, 
			       reco::TrackRefVector& tracksthis,
			       reco::TrackRefVector& Excl ) const {
  
  //std::cout<<" NEW1 Merge/Split schema "<<jetSplitMerge_<<std::endl;

  tracksthis = reco::JetTracksAssociation::getValue(jtV0,fJet);

  if(jetSplitMerge_<0) return;

  typedef std::vector<reco::JetBaseRef>::iterator JetBaseRefIterator;
  std::vector<reco::JetBaseRef> theJets = reco::JetTracksAssociation::allJets(jtV0);

  reco::TrackRefVector tracks = tracksthis;
  tracksthis.clear();

  //std::cout<<" Size of initial vector "<<tracks.size()<<" "<<fJet.et()<<" "<<fJet.eta()<<" "<<fJet.phi()<<std::endl;

  int tr=0;

  for(reco::TrackRefVector::iterator it = tracks.begin(); it != tracks.end(); it++ )
    {

      double dR2this = deltaR2 (fJet.eta(), fJet.phi(), (**it).eta(), (**it).phi());
//       double dfi = fabs(fJet.phi()-(**it).phi());
//       if(dfi>4.*atan(1.))dfi = 8.*atan(1.)-dfi;
//       double deta = fJet.eta() - (**it).eta();
//       double dR2check = sqrt(dfi*dfi+deta*deta);
      
      double scalethis = dR2this;
      if(jetSplitMerge_ == 0) scalethis = 1./fJet.et();
      if(jetSplitMerge_ == 2) scalethis = dR2this/fJet.et();
      tr++;
      int flag = 1;
      for(JetBaseRefIterator ii = theJets.begin(); ii != theJets.end(); ii++)
	{
	  if(&(**ii) == &fJet ) {continue;}
          double dR2 = deltaR2 ((*ii)->eta(), (*ii)->phi(), (**it).eta(), (**it).phi());
          double scale = dR2;
          if(jetSplitMerge_ == 0) scale = 1./fJet.et();
          if(jetSplitMerge_ == 2) scale = dR2/fJet.et();
          if(scale < scalethis) flag = 0;

          if(flag == 0) {
	    //std::cout<<" Track belong to another jet also "<<dR2<<" "<<
	    //(*ii)->et()<<" "<<(*ii)->eta()<<" "<< (*ii)->phi()<<" Track "<<(**it).eta()<<" "<<(**it).phi()<<" "<<scalethis<<" "<<scale<<" "<<flag<<std::endl;
	    break;
          }
	}

      //std::cout<<" Track "<<tr<<" "<<flag<<" "<<dR2this<<" "<<dR2check<<" Jet "<<fJet.eta()<<" "<< fJet.phi()<<" Track "<<(**it).eta()<<" "<<(**it).phi()<<std::endl;
      if(flag == 1) {tracksthis.push_back (*it);}else{Excl.push_back (*it);}
    }

  //std::cout<<" The new size of tracks "<<tracksthis.size()<<" Excludede "<<Excl.size()<<std::endl;
  return;
  
}

// -----------------------------------------------------------------------------
//
void JetPlusTrackCorrector::excludeJta( const reco::Jet& fJet, 
			       const JetTracksAssociations& jtV0, 
			       reco::TrackRefVector& tracksthis,
			       const reco::TrackRefVector& Excl ) const {
  
  //std::cout<<" NEW2" << std::endl;

  tracksthis = reco::JetTracksAssociation::getValue(jtV0,fJet);
  if(Excl.size() == 0) return;
  if(jetSplitMerge_<0) return;

  reco::TrackRefVector tracks = tracksthis;
  tracksthis.clear();
  
  //std::cout<<" Size of initial vector "<<tracks.size()<<" "<<fJet.et()<<" "<<fJet.eta()<<" "<<fJet.phi()<<std::endl;

  for(reco::TrackRefVector::iterator it = tracks.begin(); it != tracks.end(); it++ )
    {

      //std::cout<<" Track at calo surface "
      //<<" Track "<<(**it).eta()<<" "<<(**it).phi()<<std::endl;
      reco::TrackRefVector::iterator itold = find(Excl.begin(),Excl.end(),(*it));
      if(itold == Excl.end()) {
	tracksthis.push_back (*it);
      } 
      //else { std::cout<<"Exclude "<<(**it).eta()<<" "<<(**it).phi()<<std::endl; }

    }

  //std::cout<<" Size of calo tracks "<<tracksthis.size()<<std::endl;

  return;

}

// -----------------------------------------------------------------------------
//
void JetPlusTrackCorrector::particles( const AssociatedTracks& associated_tracks, 
			      const edm::Event& event, 
			      const edm::EventSetup& setup,
			      ParticleTracks& pions, 
			      ParticleTracks& muons,
			      ParticleTracks& electrons ) const { 
  
  // Some init  
  pions.clear(); 
  muons.clear(); 
  electrons.clear(); 

  // Get muons
  bool found_reco_muons = true;
  bool found_pat_muons  = true;
  edm::Handle<RecoMuons> reco_muons;
  edm::Handle<PatMuons> pat_muons;
  if ( useMuons_ ) { 

    if ( !usePat_ ) {

      // Get RECO muons
      event.getByLabel( muons_, reco_muons ); 
      if ( !reco_muons.isValid() || reco_muons.failedToGet() ) {
	found_reco_muons = false;
	edm::LogError("JetPlusTrackCorrector")
	  << "[JetPlusTrackCorrector::" << __func__ << "]"
	  << " Invalid handle to reco::GsfMuon collection"
	  << " with InputTag (label:instance:process) \"" 
	  << muons_.label() << ":"
	  << muons_.instance() << ":"
	  << muons_.process() << "\"";
      }

    } else { 
      
      // Get PAT muons
      event.getByLabel( muons_, pat_muons ); 
      if ( !pat_muons.isValid() || pat_muons.failedToGet() ) {
	found_pat_muons = false;
	edm::LogError("JetPlusTrackCorrector")
	  << "[JetPlusTrackCorrector::" << __func__ << "]"
	  << " Invalid handle to pat::Muon collection"
	  << " with InputTag (label:instance:process) \"" 
	  << muons_.label() << ":"
	  << muons_.instance() << ":"
	  << muons_.process() << "\"";
      } 

    }

  } 
  
  // Get electrons
  bool found_reco_electrons    = true;
  bool found_reco_electron_ids = true;
  bool found_pat_electrons     = true;
  edm::Handle<RecoElectrons> reco_electrons;
  edm::Handle<RecoElectronIDs> reco_electron_ids;
  edm::Handle<PatElectrons> pat_electrons;
  if ( useElectrons_ ) { 

    if ( !usePat_ ) {

      // Get RECO electrons
      event.getByLabel( electrons_, reco_electrons ); 
      if ( !reco_electrons.isValid() || reco_electrons.failedToGet() ) {
	found_reco_electrons = false;
	edm::LogError("JetPlusTrackCorrector")
	  << "[JetPlusTrackCorrector::" << __func__ << "]"
	  << " Invalid handle to reco::GsfElectron collection"
	  << " with InputTag (label:instance:process) \"" 
	  << electrons_.label() << ":"
	  << electrons_.instance() << ":"
	  << electrons_.process() << "\"";
      }

      // Get RECO electron IDs
      event.getByLabel( electronIds_, reco_electron_ids ); 
      if ( !reco_electron_ids.isValid() || reco_electron_ids.failedToGet() ) {
	found_reco_electron_ids = false;
	edm::LogError("JetPlusTrackCorrector")
	  << "[JetPlusTrackCorrector::" << __func__ << "]"
	  << " Invalid handle to reco::GsfElectron collection"
	  << " with InputTag (label:instance:process) \"" 
	  << electronIds_.label() << ":"
	  << electronIds_.instance() << ":"
	  << electronIds_.process() << "\"";
      }

    } else { 
      
      // Get PAT electrons
      event.getByLabel( electrons_, pat_electrons ); 
      if ( !pat_electrons.isValid() || pat_electrons.failedToGet() ) {
	found_pat_electrons = false;
	edm::LogError("JetPlusTrackCorrector")
	  << "[JetPlusTrackCorrector::" << __func__ << "]"
	  << " Invalid handle to pat::Electron collection"
	  << " with InputTag (label:instance:process) \"" 
	  << electrons_.label() << ":"
	  << electrons_.instance() << ":"
	  << electrons_.process() << "\"";
      }

    }

  } 
  
  // Check
  bool found_reco = found_reco_muons && found_reco_electrons && found_reco_electron_ids;
  bool found_pat  = found_pat_muons && found_pat_electrons;
  if ( !found_reco && !found_pat ) {
    edm::LogError("JetPlusTrackCorrector")
      << "[JetPlusTrackCorrector::" << __func__ << "]"
      << " Unable to access RECO or PAT collections for muons and electrons";
    return;
  }
  
  // Loop through tracks at "Vertex"
  {
    reco::TrackRefVector::const_iterator itrk = associated_tracks.atVertex_.begin();
    reco::TrackRefVector::const_iterator jtrk = associated_tracks.atVertex_.end();
    for ( ; itrk != jtrk; ++itrk ) {

      if ( useTrackQuality_ && !(*itrk)->quality(trackQuality_) ) { continue; }
      
      reco::TrackRefVector::iterator it = find( associated_tracks.atCaloFace_.begin(),
						associated_tracks.atCaloFace_.end(),
						*itrk );
      
      bool is_muon = false;
      bool is_ele  = false;
      if ( !usePat_ && found_reco ) { 
	is_muon = useMuons_     && matching( itrk, reco_muons );
	is_ele  = useElectrons_ && matching( itrk, reco_electrons, reco_electron_ids );
      } else if ( usePat_ && found_pat ) { 
	is_muon = useMuons_     && matching( itrk, pat_muons ); 
	is_ele  = useElectrons_ && matching( itrk, pat_electrons );
      }

      //@@ bug? 
      if ( it != associated_tracks.atCaloFace_.end() ) { 
	if ( is_muon )     { muons.inVertexInCalo_.push_back(*it); }
	else if ( is_ele ) { electrons.inVertexInCalo_.push_back(*it); } 
	else               { pions.inVertexInCalo_.push_back(*it); } 
      } else { 
	if ( is_muon )     { muons.inVertexOutOfCalo_.push_back(*itrk); }
	//else if ( is_ele ) { electrons.inVertexOutOfCalo_.push_back(*itrk); } 
	else               { pions.inVertexOutOfCalo_.push_back(*itrk); }
      } 
      
    } 
    
  }
  
  // Loop through tracks at "CaloFace"
  {
    reco::TrackRefVector::iterator itrk = associated_tracks.atCaloFace_.begin(); 
    reco::TrackRefVector::iterator jtrk = associated_tracks.atCaloFace_.end(); 

    for ( ; itrk != jtrk; ++itrk ) {
      
      if ( useTrackQuality_ && !(*itrk)->quality(trackQuality_) ) { continue; }
      
      //@@ bug?
      if( !pions.inVertexInCalo_.empty() ) { //||
	//!muons.inVertexInCalo_.empty() || 
	//!electrons.inVertexInCalo_.empty() ) { 
	
	reco::TrackRefVector::iterator it = find( pions.inVertexInCalo_.begin(),
						  pions.inVertexInCalo_.end(),
						  *itrk );
	
	reco::TrackRefVector::iterator im = find( muons.inVertexInCalo_.begin(),
						  muons.inVertexInCalo_.end(),
						  *itrk );
	
	//@@ bug?
	// 	reco::TrackRefVector::iterator ie = find( electrons.inVertexInCalo_.begin(),
	// 						  electrons.inVertexInCalo_.end(),
	// 						  *itrk );

	//@@ bug?	
	if ( it == pions.inVertexInCalo_.end() && 
	     im == muons.inVertexInCalo_.end() ) { //&&
	  //ie == electrons.inVertexInCalo_.end() ) {

	  //@@ bug?
	  bool is_muon = false;
	  bool is_ele  = false;
	  if ( !usePat_ && found_reco ) { 
	    is_muon = useMuons_ && matching( itrk, reco_muons );
	    is_ele  = false; //useElectrons_ && matching( itrk, reco_electrons, reco_electron_ids );
	  } else if ( usePat_ && found_pat ) { 
	    is_muon = useMuons_ && matching( itrk, pat_muons ); 
	    is_ele  = false; //useElectrons_ && matching( itrk, pat_electrons );
	  }
	  
	  if ( is_muon )     { muons.outOfVertexInCalo_.push_back(*itrk); } 
	  else if ( is_ele ) { electrons.outOfVertexInCalo_.push_back(*itrk); } //@@ bug?
	  else               { pions.outOfVertexInCalo_.push_back(*itrk); }
	  
	}
      }
    } 
    
  }
  
  if ( verbose_ && edm::isDebugEnabled() ) {
    std::stringstream ss;
    ss << "[JetPlusTrackCorrector::" << __func__ << "] Number of tracks:" << std::endl 
       << " In-cone at Vertex and in-cone at CaloFace:" << std::endl  
       << "  Pions      : " << pions.inVertexInCalo_.size() << std::endl
       << "  Muons      : " << muons.inVertexInCalo_.size() << std::endl
       << "  Electrons  : " << electrons.inVertexInCalo_.size() << std::endl
       << " In-cone at Vertex and out-of-cone at CaloFace:" << std::endl  
       << "  Pions      : " << pions.inVertexOutOfCalo_.size() << std::endl
       << "  Muons      : " << muons.inVertexOutOfCalo_.size() << std::endl
       << "  Electrons  : " << electrons.inVertexOutOfCalo_.size() << std::endl
       << " Out-of-cone at Vertex and in-cone at CaloFace:" << std::endl  
       << "  Pions      : " << pions.outOfVertexInCalo_.size() << std::endl
       << "  Muons      : " << muons.outOfVertexInCalo_.size() << std::endl
       << "  Electrons  : " << electrons.outOfVertexInCalo_.size() << std::endl;
    LogTrace("JetPlusTrackCorrector") << ss.str();
  }
  
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::matching( reco::TrackRefVector::const_iterator itrk, 
			     const edm::Handle<RecoMuons>& muons ) const {
  
  if ( muons->empty() ) { return false; }

  RecoMuons::const_iterator imuon = muons->begin(); 
  RecoMuons::const_iterator jmuon = muons->end(); 
  for ( ; imuon != jmuon; ++imuon ) {
    
    if ( imuon->innerTrack().isNull() ||
	 !muon::isGoodMuon(*imuon,muon::TMLastStationTight) ||
	 imuon->innerTrack()->pt() < 3.0 ) { continue; }
    
    if ( itrk->id() != imuon->innerTrack().id() ) {
      edm::LogError("JetPlusTrackCorrector")
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
bool JetPlusTrackCorrector::matching( reco::TrackRefVector::const_iterator itrk, 
			     const edm::Handle<PatMuons>& muons ) const {
  
  if ( muons->empty() ) { return false; }
  
  PatMuons::const_iterator imuon = muons->begin(); 
  PatMuons::const_iterator jmuon = muons->end(); 
  for ( ; imuon != jmuon; ++imuon ) {

    if ( imuon->innerTrack().isNull() ||
	 !muon::isGoodMuon(*imuon,muon::TMLastStationTight) ||
	 imuon->innerTrack()->pt() < 3.0 ) { continue; }
    
    if ( itrk->id() != imuon->innerTrack().id() ) {
      edm::LogError("JetPlusTrackCorrector")
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
bool JetPlusTrackCorrector::matching( reco::TrackRefVector::const_iterator itrk, 
			     const edm::Handle<RecoElectrons>& electrons,
			     const edm::Handle<RecoElectronIDs>& electron_ids ) const {

  if ( electrons->empty() ) { return false; }
  
  double deltaR = 999.;
  double deltaRMIN = 999.;
	
  uint32_t electron_index = 0;
  RecoElectrons::const_iterator ielec = electrons->begin(); 
  RecoElectrons::const_iterator jelec = electrons->end(); 
  for ( ; ielec != jelec; ++ielec ) {
    
    edm::Ref<RecoElectrons> electron_ref( electrons, electron_index );
    electron_index++;
    
    if ( (*electron_ids)[electron_ref] < 1.e-6 ) { continue; } //@@ Check for null value 
    
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

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::matching( reco::TrackRefVector::const_iterator itrk, 
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

// -----------------------------------------------------------------------------
//
double JetPlusTrackCorrector::correction( const reco::TrackRefVector& tracks, 
				 ParticleResponse& track_response,
				 bool in_cone_at_vertex,
				 bool in_cone_at_calo_face,
				 double mass, 
				 double mip ) const { 

  // Correction to be applied
  double correction = 0.;
  
  // Clear and reset the response matrix
  track_response.clear();
  track_response.resize( response().nEtaBins(), 
			 response().nPtBins() );

  // Iterate through tracks
  if ( !tracks.empty() ) {
    reco::TrackRefVector::iterator itrk = tracks.begin();
    reco::TrackRefVector::iterator jtrk = tracks.end();
    for ( ; itrk != jtrk; ++itrk ) {

      // Ignore high-pt tracks 
      if ( in_cone_at_calo_face && //@@ only when in-cone
	   mip < 0. &&          //@@ only when not a mip
	   (*itrk)->pt() >= 50. ) { continue; }
      
      // Track momentum
      double momentum = sqrt( (*itrk)->px() * (*itrk)->px() + 
			      (*itrk)->py() * (*itrk)->py() + 
			      (*itrk)->pz() * (*itrk)->pz() + 
			      mass * mass );
      
      // Add track momentum (if in-cone at vertex)
      if ( in_cone_at_vertex ) { correction += momentum; }

      // Check if particle is mip or not
      if ( mip > 0. ) { 
	if ( in_cone_at_calo_face ) { correction -= mip; }
      } else { 
	// Find appropriate eta/pt bin for given track
	for ( uint32_t ieta = 0; ieta < response().nEtaBins()-1; ++ieta ) {
	  for ( uint32_t ipt = 0; ipt < response().nPtBins()-1; ++ipt ) {
	    double eta = fabs( (*itrk)->eta() );
	    if ( eta > response().eta(ieta) && ( ieta+1 == response().nEtaBins() || eta < response().eta(ieta+1) ) ) {
	      double pt = fabs( (*itrk)->pt() );
	      if ( pt > response().pt(ipt) && ( ipt+1 == response().nPtBins() || pt < response().pt(ipt+1) ) ) {
		
		// Subtract expected response (if appropriate)
		if ( in_cone_at_calo_face ) { correction -= ( momentum * response().value(ieta,ipt) ); } 
		
		// Record track momentum for efficiency correction
		track_response.addE( ieta, ipt, momentum );
		
		// Debug
		if ( verbose_ && edm::isDebugEnabled() ) {
		  std::stringstream temp; 
		  temp << " Response[" << ieta << "," << ipt << "]";
		  std::stringstream ss;
		  ss << "[JetPlusTrackCorrector::" << __func__ << "]" << std::endl
		     << " Track eta / pt    : " << eta << " / " << pt << std::endl
		     << temp.str() << std::setw(21-temp.str().size()) << " : " 
		     << response().value(ieta,ipt) << std::endl
		     << " Track momentum    : " << momentum << std::endl
		     << " Energy subtracted : " << momentum * response().value(ieta,ipt) << std::endl
		     << " Energy correction : " << correction;
		  // 		  int k = ieta*response().nPtBins()+ipt;
		  // 		  ss << "        k eta/pT index = " << k
		  // 		     << " netracks_incone[k] = " << track_response.nTrks( ieta, ipt )
		  // 		     << " emean_incone[k] = " << track_response.sumE( ieta, ipt )
		  // 		     << " i,j "<<ieta<<" "<<ipt<<" "<<response().value(ieta,ipt)
		  // 		     << " echar "<<momentum<<" "<<response().value(ieta,ipt)*momentum;
		  LogTrace("JetPlusTrackCorrector") << ss.str();
		}
		
	      }
	    }
	  }
	}
      }

    }
  }

  return correction;

}

// -----------------------------------------------------------------------------
//
double JetPlusTrackCorrector::correction( ParticleResponse& track_response,
				 bool in_cone_at_calo_face ) const { 
  
  // Correction to be applied
  double correction = 0.;
  
  // Iterate through eta/pt bins
  for ( uint32_t ieta = 0; ieta < response().nEtaBins()-1; ++ieta ) {
    for ( uint32_t ipt = 0; ipt < response().nPtBins()-1; ++ipt ) {
      uint16_t ntrks = track_response.nTrks(ieta,ipt);
      if ( !ntrks ) { continue; }
      double mean  = track_response.meanE(ieta,ipt);
      double eff   = ( 1. - efficiency().value(ieta,ipt) ) / efficiency().value(ieta,ipt);
      double corr  = ntrks * eff * mean;
      correction  += corr;
      if ( in_cone_at_calo_face ) { correction -= corr * leakage().value(ieta,ipt) * response().value(ieta,ipt); }
    }
  }
  
  return correction;

}


// -----------------------------------------------------------------------------
//
Map::Map( std::string input, bool verbose )
  : eta_(),
    pt_(),
    data_()
{ 

  // Some init
  clear();
  std::vector<Element> data;

  // Parse file
  std::string file = edm::FileInPath(input).fullPath();
  std::ifstream in( file.c_str() );
  string line;
  uint32_t ieta_old = 0; 
  while ( std::getline( in, line ) ) {
    if ( !line.size() || line[0]=='#' ) { continue; }
    std::istringstream ss(line);
    Element temp;
    ss >> temp.ieta_ >> temp.ipt_ >> temp.eta_ >> temp.pt_ >> temp.val_;
    data.push_back(temp);
    if ( !ieta_old || temp.ieta_ != ieta_old ) { 
      if ( eta_.size() < temp.ieta_+1 ) { eta_.resize(temp.ieta_+1,0.); }
      eta_[temp.ieta_] = temp.eta_;
      ieta_old = temp.ieta_;
    }
    if ( pt_.size() < temp.ipt_+1 ) { pt_.resize(temp.ipt_+1,0.); }
    pt_[temp.ipt_] = temp.pt_;
  }
  
  // Populate container
  data_.resize( eta_.size(), VDouble( pt_.size(), 0. ) );
  std::vector<Element>::const_iterator idata = data.begin();
  std::vector<Element>::const_iterator jdata = data.end();
  for ( ; idata != jdata; ++idata ) { data_[idata->ieta_][idata->ipt_] = idata->val_; }

  // Check
  if ( data_.empty() || data_[0].empty() ) {
    std::stringstream ss;
    ss << "[jpt::Map::Map]"
       << " Problem parsing map in location \"" 
       << file << "\"! ";
    edm::LogError("JetPlusTrackCorrector") << ss.str();
  }

  // Check
  if ( eta_.size() != data_.size() || 
       pt_.size() != ( data_.empty() ? 0 : data_[0].size() ) ) {
    std::stringstream ss;
    ss << "[jpt::Map::Map]"
       << " Discrepancy b/w number of bins!";
    edm::LogError("JetPlusTrackCorrector") << ss.str();
  }

  // Debug
  if ( verbose && edm::isDebugEnabled() ) {
    std::stringstream ss;
    ss << "[jpt::Map::Map]"
       << " Parsed contents of map at location:" << std::endl
       << "\"" << file << "\"" << std::endl
       << " Number of bins in eta : " << data_.size() << std::endl 
       << " Number of bins in pt  : " << ( data_.empty() ? 0 : data_[0].size() ) << std::endl;
    VVDouble::const_iterator ieta = data_.begin();
    VVDouble::const_iterator jeta = data_.end();
    for ( ; ieta != jeta; ++ieta ) {
      VDouble::const_iterator ipt = ieta->begin();
      VDouble::const_iterator jpt = ieta->end();
      for ( ; ipt != jpt; ++ipt ) {
	uint32_t eta_bin = static_cast<uint32_t>( ieta - data_.begin() );
	uint32_t pt_bin  = static_cast<uint32_t>( ipt - ieta->begin() );
	ss << " EtaBinNumber: " << eta_bin 
	   << " PtBinNumber: " << pt_bin 
	   << " EtaValue: " << eta_[ eta_bin ]
	   << " PtValue: " << pt_[ pt_bin ]
	   << " Value: " << data_[eta_bin][pt_bin]
	   << std::endl;
      }
    }
    LogTrace("JetPlusTrackCorrector") << ss.str();
  }
  
}

// -----------------------------------------------------------------------------
//
Map::Map() 
  : eta_(),
    pt_(),
    data_()
{ 
  clear();
}

// -----------------------------------------------------------------------------
//
Map::~Map() {
  clear();
}

// -----------------------------------------------------------------------------
//
void Map::clear() {
  eta_.clear();
  pt_.clear();
  data_.clear();
}
 
// -----------------------------------------------------------------------------
//
double Map::eta( uint32_t eta_bin ) const {
  if ( eta_bin < eta_.size() ) { return eta_[eta_bin]; }
  else { 
    edm::LogWarning("JetPlusTrackCorrector") 
      << "[jpt::Map::eta]"
      << " Trying to access element " << eta_bin
      << " of a vector with size " << eta_.size()
      << "!";
    return -1.; 
  }
}

// -----------------------------------------------------------------------------
//
double Map::pt( uint32_t pt_bin ) const {
  if ( pt_bin < pt_.size() ) { return pt_[pt_bin]; }
  else { 
    edm::LogWarning("JetPlusTrackCorrector") 
      << "[jpt::Map::pt]"
      << " Trying to access element " << pt_bin
      << " of a vector with size " << pt_.size()
      << "!";
    return -1.; 
  }
}

// -----------------------------------------------------------------------------
//
double Map::value( uint32_t eta_bin, uint32_t pt_bin ) const {
  if ( eta_bin < data_.size() && 
       pt_bin < ( data_.empty() ? 0 : data_[0].size() ) ) { return data_[eta_bin][pt_bin]; }
  else { 
    edm::LogWarning("JetPlusTrackCorrector") 
      << "[jpt::Map::value]"
      << " Trying to access element (" << eta_bin << "," << pt_bin << ")"
      << " of a vector with size (" << data_.size() << "," << ( data_.empty() ? 0 : data_[0].size() ) << ")"
      << "!";
    return 1.; 
  }
}
 
// -----------------------------------------------------------------------------
//
ParticleTracks::ParticleTracks() 
  : inVertexInCalo_(),
    outOfVertexInCalo_(),
    inVertexOutOfCalo_()
{ 
  clear();
}

// -----------------------------------------------------------------------------
//
ParticleTracks::~ParticleTracks() {
  clear();
}

// -----------------------------------------------------------------------------
//
void ParticleTracks::clear() {
  inVertexInCalo_.clear();
  outOfVertexInCalo_.clear();
  inVertexOutOfCalo_.clear();
}
 
// -----------------------------------------------------------------------------
//
AssociatedTracks::AssociatedTracks() 
  : atVertex_(),
    atCaloFace_()
{ 
  clear();
}

// -----------------------------------------------------------------------------
//
AssociatedTracks::~AssociatedTracks() {
  clear();
}

// -----------------------------------------------------------------------------
//
void AssociatedTracks::clear() {
  atVertex_.clear();
  atCaloFace_.clear();
}

// -----------------------------------------------------------------------------
//
uint16_t ParticleResponse::nTrks( uint32_t eta_bin, uint32_t pt_bin ) const {
  if ( check(eta_bin,pt_bin) ) { 
    return data_[eta_bin][pt_bin].first; 
  } else { return 0; }
}

// -----------------------------------------------------------------------------
//
double ParticleResponse::sumE( uint32_t eta_bin, uint32_t pt_bin ) const {
  if ( check(eta_bin,pt_bin) ) { 
    return data_[eta_bin][pt_bin].second; 
  } else { return 0.; }
}

// -----------------------------------------------------------------------------
//
double ParticleResponse::meanE( uint32_t eta_bin, uint32_t pt_bin ) const {
  if ( check(eta_bin,pt_bin) ) { 
    Pair tmp = data_[eta_bin][pt_bin]; 
    if ( tmp.first ) { return tmp.second / tmp.first; }
    else { return 0.; }
  } else { return 0.; }
}

// -----------------------------------------------------------------------------
//
void ParticleResponse::addE( uint32_t eta_bin, uint32_t pt_bin, double energy ) {
  if ( check(eta_bin,pt_bin) ) { 
    data_[eta_bin][pt_bin].first++; 
    data_[eta_bin][pt_bin].second += energy;
  } 
}

// -----------------------------------------------------------------------------
//
bool ParticleResponse::check( uint32_t eta_bin, uint32_t pt_bin ) const {
  if ( eta_bin < data_.size() && pt_bin < ( data_.empty() ? 0 : data_[0].size() ) ) { return true; }
  else { 
    edm::LogWarning("JetPlusTrackCorrector") 
      << "[jpt::ParticleResponse::check]"
      << " Trying to access element (" << eta_bin << "," << pt_bin << ")"
      << " of a vector with size (" << data_.size() << "," << ( data_.empty() ? 0 : data_[0].size() ) << ")"
      << "!";
    return false; 
  }
}

// -----------------------------------------------------------------------------
//
void ParticleResponse::clear() { 
  data_.clear();
}

// -----------------------------------------------------------------------------
//
void ParticleResponse::resize( uint32_t eta_bins, uint32_t pt_bins, Pair value ) {
  data_.resize( eta_bins, VPair( pt_bins, value ) );
}
