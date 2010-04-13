#ifndef PhysicsTools_PatUtils_interface_JetIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_JetIDSelectionFunctor_h


/**
  \class    JetIDSelectionFunctor JetIDSelectionFunctor.h "PhysicsTools/Utilities/interface/JetIDSelectionFunctor.h"
  \brief    Jet selector for pat::Jets and for CaloJets

  Selector functor for pat::Jets that implements quality cuts based on
  studies of noise patterns. 

  Please see https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATSelectors
  for a general overview of the selectors. 

  \author Salvatore Rappoccio (Update: Amnon Harel)
  \version  $Id: JetIDSelectionFunctor.h,v 1.5 2010/02/10 20:06:25 srappocc Exp $
*/




#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"

#include <TMath.h>
class JetIDSelectionFunctor : public Selector<pat::Jet>  {

 public: // interface

  enum Version_t { CRAFT08, PURE09, DQM09, N_VERSIONS };
  enum Quality_t { MINIMAL, LOOSE_AOD, LOOSE, TIGHT, N_QUALITY};
  

 JetIDSelectionFunctor( Version_t version, Quality_t quality ) :
  version_(version), quality_(quality)
  {

    push_back("MINIMAL_EMF");

    push_back("LOOSE_AOD_fHPD");
    push_back("LOOSE_AOD_N90Hits");
    push_back("LOOSE_AOD_EMF");

    push_back("LOOSE_fHPD");
    push_back("LOOSE_N90Hits");
    push_back("LOOSE_EMF");

    push_back("TIGHT_fHPD");
    push_back("TIGHT_EMF");

    push_back("LOOSE_nHit");
    push_back("LOOSE_als");
    push_back("LOOSE_fls");
    push_back("LOOSE_foot");

    push_back("TIGHT_nHit");
    push_back("TIGHT_als");
    push_back("TIGHT_fls");
    push_back("TIGHT_foot");
    push_back("widths");
    push_back("EF_N90Hits");
    push_back("EF_EMF");


    bool use_09_fwd_id = version_ != CRAFT08; // CRAFT08 predates the 09 forward ID cuts
    bool use_dqm_09 = version_ == DQM09 && quality_ != LOOSE_AOD;

    // all appropriate for version and format (AOD complications) are on by default
    set( "MINIMAL_EMF" );
    set( "LOOSE_AOD_fHPD" );
    set( "LOOSE_AOD_N90Hits" );
    set( "LOOSE_AOD_EMF", ! use_09_fwd_id ); // except in CRAFT08, this devolves into MINIMAL_EMF
    set( "LOOSE_fHPD" );
    set( "LOOSE_N90Hits" );
    set( "LOOSE_EMF", ! use_09_fwd_id ); // except in CRAFT08, this devolves into MINIMAL_EMF
    set( "TIGHT_fHPD" );
    set( "TIGHT_EMF" );

    set( "LOOSE_nHit", use_09_fwd_id );
    set( "LOOSE_als", use_09_fwd_id );
    set( "TIGHT_nHit", use_09_fwd_id );
    set( "TIGHT_als", use_09_fwd_id );
    set( "widths", use_09_fwd_id );
    set( "EF_N90Hits", use_09_fwd_id );
    set( "EF_EMF", use_09_fwd_id );

    set( "LOOSE_fls", use_dqm_09 );
    set( "LOOSE_foot", use_dqm_09 );
    set( "TIGHT_fls", use_dqm_09 );
    set( "TIGHT_foot", use_dqm_09 );

    // now set the return values for the ignored parts
    bool use_loose_aod = false;
    bool use_loose = false;
    bool use_tight = false;
    bool use_tight_09_fwd_id = false;
    bool use_loose_09_fwd_id = false;
    // if ( quality_ == MINIMAL ) nothing to do...
    if ( quality_ == LOOSE ) {
      use_loose = true;
      if( use_09_fwd_id ) use_loose_09_fwd_id = true;
    } 
    if ( quality_ == LOOSE_AOD ) {
      use_loose_aod = true;
      if( use_09_fwd_id ) use_loose_09_fwd_id = true;
    }
    if ( quality_ == TIGHT ) {
      use_tight = true;
      if( use_09_fwd_id ) use_tight_09_fwd_id = true;
    }

    if( ! use_loose_aod ) {
      set("LOOSE_AOD_fHPD", false );
      set("LOOSE_AOD_N90Hits", false );
      set("LOOSE_AOD_EMF", false );
    }

    if( ! ( use_loose || use_tight ) ) { // the CRAFT08 cuts are cumulative
      set("LOOSE_N90Hits", false );
      set("LOOSE_fHPD", false );
      set("LOOSE_EMF", false );
    }

    if( ! use_tight ) {
      set("TIGHT_fHPD", false );
      set("TIGHT_EMF", false );      
    }

    if( ! use_loose_09_fwd_id ) { // the FWD09 cuts are not
      set( "LOOSE_nHit", false );
      set( "LOOSE_als", false );
      if( use_dqm_09 ) {
	set( "LOOSE_fls", false );
	set( "LOOSE_foot", false );
      }
    } // not using loose 09 fwd ID

    if( ! use_tight_09_fwd_id ) {
      set( "TIGHT_nHit", false );
      set( "TIGHT_als", false );
      set( "widths", false );
      set( "EF_N90Hits", false );
      set( "EF_EMF", false );
      if( use_dqm_09 ) {
	set( "TIGHT_fls", false );
	set( "TIGHT_foot", false );
      }
    } // not using tight 09 fwd ID

    retInternal_ = getBitTemplate();
  }

  // this functionality should be migrated into JetIDHelper in future releases
  unsigned int count_hits( const std::vector<CaloTowerPtr> & towers )
  {
    unsigned int nHit = 0;
    for ( unsigned int iTower = 0; iTower < towers.size() ; ++iTower ) {
      const vector<DetId>& cellIDs = towers[iTower]->constituents();  // cell == recHit
      nHit += cellIDs.size();
    }
    return nHit;
  }

  // 
  // Accessor from PAT jets
  // 
  bool operator()( const pat::Jet & jet, std::strbitset & ret )  
  {
    if ( ! jet.isCaloJet() ) {
      edm::LogWarning( "NYI" )<<"Criteria for pat::Jet-s other than CaloJets are not yet implemented";
      return false;
    }
    if ( version_ == CRAFT08 ) return craft08Cuts( jet.p4(), jet.emEnergyFraction(), jet.jetID(), ret );
    if ( version_ == PURE09 || version_ == DQM09 ) {
      unsigned int nHit = count_hits( jet.getCaloConstituents() );
      return fwd09Cuts( jet.p4(), jet.emEnergyFraction(), jet.etaetaMoment(), jet.phiphiMoment(), nHit,
			jet.jetID(), ret );
    }
    edm::LogWarning( "BadInput | NYI" )<<"Requested version ("<<version_<<") is unknown";
    return false;
  }

  // accessor from PAT jets without the ret
  using Selector<pat::Jet>::operator();

  // 
  // Accessor from *CORRECTED* 4-vector, EMF, and Jet ID. 
  // This can be used with reco quantities. 
  // 
  bool operator()( reco::Candidate::LorentzVector const & correctedP4, 
		   double emEnergyFraction, 
		   reco::JetID const & jetID,
		   std::strbitset & ret )  
  {
    if ( version_ == CRAFT08 ) return craft08Cuts( correctedP4, emEnergyFraction, jetID, ret );
    edm::LogWarning( "BadInput | NYI" )<<"Requested version ("<<version_
				       <<") is unknown or doesn't match the depreceated interface used";
    return false;
  }
  /// accessor like previous, without the ret
  virtual bool operator()( reco::Candidate::LorentzVector const & correctedP4, 
			   double emEnergyFraction, 
			   reco::JetID const & jetID )
  {
    retInternal_.set(false);
    operator()(correctedP4,emEnergyFraction, jetID, retInternal_);
    setIgnored(retInternal_);
    return (bool)retInternal_;
  }

  // 
  // Accessor from CaloJet and Jet ID. Jets MUST BE corrected for craft08, but uncorrected for later cuts...
  // This can be used with reco quantities. 
  // 
  bool operator()( reco::CaloJet const & jet,
		   reco::JetID const & jetID,
		   std::strbitset & ret )  
  {
    if ( version_ == CRAFT08 ) return craft08Cuts( jet.p4(), jet.emEnergyFraction(), jetID, ret );
    if ( version_ == PURE09 || version_ == DQM09 ) {
      unsigned int nHit = count_hits( jet.getCaloConstituents() );
      return fwd09Cuts( jet.p4(), jet.emEnergyFraction(), jet.etaetaMoment(), jet.phiphiMoment(), nHit,
			jetID, ret );
    }
    edm::LogWarning( "BadInput | NYI" )<<"Requested version ("<<version_<<") is unknown";
    return false;
  }

  /// accessor like previous, without the ret
  virtual bool operator()( reco::CaloJet const & jet,
			   reco::JetID const & jetID )
  {
    retInternal_.set(false);
    operator()(jet, jetID, retInternal_);
    setIgnored(retInternal_);
    return (bool)retInternal_;
  }
  
  // 
  // cuts based on craft 08 analysis. 
  // 
  bool craft08Cuts( reco::Candidate::LorentzVector const & correctedP4, 
		    double emEnergyFraction,
		    reco::JetID const & jetID,
		    std::strbitset & ret) 
  {
    
    ret.set(false);

    // cache some variables
    double abs_eta = TMath::Abs( correctedP4.eta() );
    double corrPt = correctedP4.pt();
    double emf = emEnergyFraction;

    if ( ignoreCut("MINIMAL_EMF") || abs_eta > 2.6 || emf > 0.01 ) passCut( ret, "MINIMAL_EMF");
            
    if ( quality_ == LOOSE_AOD ) {
      // loose fhpd cut from aod
      if ( ignoreCut("LOOSE_AOD_fHPD")    || jetID.approximatefHPD < 0.98 ) passCut( ret, "LOOSE_AOD_fHPD");
      // loose n90 hits cut
      if ( ignoreCut("LOOSE_AOD_N90Hits") || jetID.hitsInN90 > 1 ) passCut( ret, "LOOSE_AOD_N90Hits");
      
      // loose EMF Cut from aod
      bool emf_loose = true;
      if( abs_eta <= 2.6 ) { // HBHE
	if( emEnergyFraction <= 0.01 ) emf_loose = false;
      } else {                // HF
	if( emEnergyFraction <= -0.9 ) emf_loose = false;
	if( corrPt > 80 && emEnergyFraction >= 1 ) emf_loose = false;
      }
      if ( ignoreCut("LOOSE_AOD_EMF") || emf_loose ) passCut(ret, "LOOSE_AOD_EMF");
	
    }
    else if ( quality_ == LOOSE || quality_ == TIGHT ) {
      // loose fhpd cut
      if ( ignoreCut("LOOSE_fHPD")    || jetID.fHPD < 0.98 ) passCut( ret, "LOOSE_fHPD");
      // loose n90 hits cut
      if ( ignoreCut("LOOSE_N90Hits") || jetID.n90Hits > 1 ) passCut( ret, "LOOSE_N90Hits");

      // loose EMF Cut
      bool emf_loose = true;
      if( abs_eta <= 2.6 ) { // HBHE
	if( emEnergyFraction <= 0.01 ) emf_loose = false;
      } else {                // HF
	if( emEnergyFraction <= -0.9 ) emf_loose = false;
	if( corrPt > 80 && emEnergyFraction >= 1 ) emf_loose = false;
      }
      if ( ignoreCut("LOOSE_EMF") || emf_loose ) passCut(ret, "LOOSE_EMF");
 
      if ( quality_ == TIGHT ) {
	// tight fhpd cut
	bool tight_fhpd = true;
	if ( corrPt >= 25 && jetID.fHPD >= 0.95 ) tight_fhpd = false; // this was supposed to use raw pT, see AN2009/087 :-(
	if ( ignoreCut("TIGHT_fHPD") || tight_fhpd ) passCut(ret, "TIGHT_fHPD");
	
	// tight emf cut
	bool tight_emf = true;
	if( abs_eta >= 1 && corrPt >= 80 && emEnergyFraction >= 1 ) tight_emf = false; // outside HB	  
	if( abs_eta >= 2.6 ) { // outside HBHE
	  if( emEnergyFraction <= -0.3 ) tight_emf = false;
	  if( abs_eta < 3.25 ) { // HE-HF transition region
	    if( corrPt >= 50 && emEnergyFraction <= -0.2 ) tight_emf = false;
	    if( corrPt >= 80 && emEnergyFraction <= -0.1 ) tight_emf = false;	
	    if( corrPt >= 340 && emEnergyFraction >= 0.95 ) tight_emf = false;
	  } else { // HF
	    if( emEnergyFraction >= 0.9 ) tight_emf = false;
	    if( corrPt >= 50 && emEnergyFraction <= -0.2 ) tight_emf = false;
	    if( corrPt >= 50 && emEnergyFraction >= 0.8 ) tight_emf = false;
	    if( corrPt >= 130 && emEnergyFraction <= -0.1 ) tight_emf = false;
	    if( corrPt >= 130 && emEnergyFraction >= 0.7 ) tight_emf = false;
	    
	  } // end if HF
	}// end if outside HBHE
	if ( ignoreCut("TIGHT_EMF") || tight_emf ) passCut(ret, "TIGHT_EMF");
      }// end if tight
    }// end if loose or tight

    setIgnored( ret );    

    return (bool)ret;
  }

  // 
  // cuts based on craft 08 analysis + forward jet ID based on 09 data 
  // 
  bool fwd09Cuts( reco::Candidate::LorentzVector const & rawP4, 
		  double emEnergyFraction, double etaWidth, double phiWidth, unsigned int nHit, 
		  reco::JetID const & jetID,
		  std::strbitset & ret) 
  {
    ret.set(false);

    // cache some variables
    double abs_eta = TMath::Abs( rawP4.eta() );
    double rawPt = rawP4.pt();
    double emf = emEnergyFraction;
    double fhf = jetID.fLong + jetID.fShort;
    double lnpt = ( rawPt > 0 ) ? TMath::Log( rawPt ) : -10;
    double lnE = ( rawP4.energy() > 0 ) ? TMath::Log( rawP4.energy() ) : -10;

    bool HB = abs_eta < 1.0;
    bool EF = 2.6 <= abs_eta && abs_eta < 3.4 && 0.1 <= fhf && fhf < 0.9;
    bool HBHE = abs_eta < 2.6 || ( abs_eta < 3.4 && fhf < 0.1 );
    bool HF = 3.4 <= abs_eta  || ( 2.6 <= abs_eta && 0.9 <= fhf );
    bool HFa = HF && abs_eta < 3.8;
    bool HFb = HF && ! HFa;

    // HBHE cuts as in CRAFT08
    // - but using raw pTs
    // ========================

    if ( (!HBHE) || ignoreCut("MINIMAL_EMF") || emf > 0.01 ) passCut( ret, "MINIMAL_EMF");
            
    // loose fhpd cut from AOD
    if ( (!HBHE) || ignoreCut("LOOSE_AOD_fHPD") || jetID.approximatefHPD < 0.98 ) passCut( ret, "LOOSE_AOD_fHPD");
    // loose n90 hits cut from AOD
    if ( (!HBHE) || ignoreCut("LOOSE_AOD_N90Hits") || jetID.hitsInN90 > 1 ) passCut( ret, "LOOSE_AOD_N90Hits");

    // loose fhpd cut
    if ( (!HBHE) || ignoreCut("LOOSE_fHPD") || jetID.fHPD < 0.98 ) passCut( ret, "LOOSE_fHPD");
    // loose n90 hits cut
    if ( (!HBHE) || ignoreCut("LOOSE_N90Hits") || jetID.n90Hits > 1 ) passCut( ret, "LOOSE_N90Hits");
 
    // tight fhpd cut
    if ( (!HBHE) || ignoreCut("TIGHT_fHPD") || rawPt < 25 || jetID.fHPD < 0.95 ) passCut(ret, "TIGHT_fHPD");
      
    // tight emf cut
    if ( (!HBHE) || ignoreCut("TIGHT_EMF") || HB || rawPt < 55 || emf < 1 ) passCut(ret, "TIGHT_EMF");

 
    // EF - these cuts are only used in "tight", but there's no need for this test here.

    if( (!EF) || ignoreCut( "EF_N90Hits" ) 
	|| jetID.n90Hits > 1 + 1.5 * TMath::Max( 0., lnpt - 1.5 ) ) 
      passCut( ret, "EF_N90Hits" );

    if( (!EF) || ignoreCut( "EF_EMF" ) 
	|| emf > TMath::Max( -0.9, -0.1 - 0.05 * TMath::Power( TMath::Max( 0., 5 - lnpt ), 2. ) ) )
      passCut( ret, "EF_EMF" );

    // both EF and HF

    if( ( !( EF || HF ) ) || ignoreCut( "TIGHT_fls" )
	|| ( EF && jetID.fLS < TMath::Min( 0.8, 0.1 + 0.016 * TMath::Power( TMath::Max( 0., 6 - lnpt ), 2.5 ) ) )
	|| ( HFa && jetID.fLS < TMath::Min( 0.6, 0.05 + 0.045 * TMath::Power( TMath::Max( 0., 7.5 - lnE ), 2.2 ) ) ) 
	|| ( HFb && jetID.fLS < TMath::Min( 0.1, 0.05 + 0.07 * TMath::Power( TMath::Max( 0., 7.8 - lnE ), 2. ) ) ) )
      passCut( ret, "TIGHT_fls" );

    if( ( !( EF || HF ) ) || ignoreCut( "widths" )
	|| ( 1E-10 < etaWidth && etaWidth < 0.12 && 
	     1E-10 < phiWidth && phiWidth < 0.12 ) )
      passCut( ret, "widths" );

    // HF cuts

    if( (!HF) || ignoreCut( "LOOSE_nHit" ) 
	|| ( HFa && nHit > 1 + 2.4*( lnpt - 1. ) ) 
	|| ( HFb && nHit > 1 + 3.*( lnpt - 1. ) ) )
      passCut( ret, "LOOSE_nHit" );

    if( (!HF) || ignoreCut( "LOOSE_als" )
	|| ( emf < 0.6 + 0.05 * TMath::Power( TMath::Max( 0., 9 - lnE ), 1.5 ) &&
	     emf > -0.2 - 0.041 * TMath::Power( TMath::Max( 0., 7.5 - lnE ), 2.2 ) ) )
      passCut( ret, "LOOSE_als" ); 
      
    if( (!HF) || ignoreCut( "LOOSE_fls" )
	|| ( HFa && jetID.fLS < TMath::Min( 0.9, 0.1 + 0.05 * TMath::Power( TMath::Max( 0., 7.5 - lnE ), 2.2 ) ) )
	|| ( HFb && jetID.fLS < TMath::Min( 0.6, 0.1 + 0.065 * TMath::Power( TMath::Max( 0., 7.5 - lnE ), 2.2 ) ) ) )
      passCut( ret, "LOOSE_fls" ); 

    if( (!HF) || ignoreCut( "LOOSE_foot" )
	|| jetID.fHFOOT < 0.9 )
      passCut( ret, "LOOSE_foot" );
      
    if( (!HF) || ignoreCut( "TIGHT_nHit" )
	|| ( HFa && nHit > 1 + 2.7*( lnpt - 0.8 ) ) 
	|| ( HFb && nHit > 1 + 3.5*( lnpt - 0.8 ) ) )
      passCut( ret, "TIGHT_nHit" );

    if( (!HF) || ignoreCut( "TIGHT_als" ) 
	|| ( emf < 0.5 + 0.057 * TMath::Power( TMath::Max( 0., 9 - lnE ), 1.5 ) &&
	     emf > TMath::Max( -0.6, -0.1 - 0.026 * TMath::Power( TMath::Max( 0., 8 - lnE ), 2.2 ) ) ) )
      passCut( ret, "TIGHT_als" ); 

    if( (!HF) || ignoreCut( "TIGHT_foot" ) 
	|| jetID.fLS < 0.5 )
      passCut( ret, "TIGHT_foot" );

    setIgnored( ret );    

    return (bool)ret;
  }
  
 private: // member variables
  
  Version_t version_;
  Quality_t quality_;
  
};

#endif
