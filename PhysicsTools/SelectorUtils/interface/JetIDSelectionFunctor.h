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
  \version  $Id: JetIDSelectionFunctor.h,v 1.15 2010/08/31 20:31:50 srappocc Exp $
*/




#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TMath.h>
class JetIDSelectionFunctor : public Selector<pat::Jet>  {

 public: // interface

  enum Version_t { CRAFT08, PURE09, DQM09, N_VERSIONS };
  enum Quality_t { MINIMAL, LOOSE_AOD, LOOSE, TIGHT, N_QUALITY};

  JetIDSelectionFunctor() {}

#ifndef __GCCXML__
  JetIDSelectionFunctor( edm::ParameterSet const & parameters, edm::ConsumesCollector& iC ) :
    JetIDSelectionFunctor(parameters)
  {}
#endif


  JetIDSelectionFunctor( edm::ParameterSet const & parameters ) {
    std::string versionStr = parameters.getParameter<std::string>("version");
    std::string qualityStr = parameters.getParameter<std::string>("quality");
    Quality_t quality = N_QUALITY;

    if      ( qualityStr == "MINIMAL" )   quality = MINIMAL;
    else if ( qualityStr == "LOOSE_AOD" ) quality = LOOSE_AOD;
    else if ( qualityStr == "LOOSE" )     quality = LOOSE;
    else if ( qualityStr == "TIGHT" )     quality = TIGHT;
    else
      throw cms::Exception("InvalidInput") << "Expect quality to be one of MINIMAL, LOOSE_AOD, LOOSE,TIGHT" << std::endl;


    if ( versionStr == "CRAFT08" ) {
      initialize( CRAFT08, quality );
    }
    else if ( versionStr == "PURE09" ) {
      initialize( PURE09, quality );
    }
    else if ( versionStr == "DQM09" ) {
      initialize( DQM09, quality );
    }
    else {
      throw cms::Exception("InvalidInput") << "Expect version to be one of CRAFT08, PURE09, DQM09" << std::endl;
    }
  }

  JetIDSelectionFunctor( Version_t version, Quality_t quality ) {
    initialize(version, quality);
  }

 void initialize( Version_t version, Quality_t quality )
  {

    version_ = version;
    quality_ = quality;

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




    index_MINIMAL_EMF_       = index_type(&bits_, "MINIMAL_EMF");

    index_LOOSE_AOD_fHPD_    = index_type(&bits_, "LOOSE_AOD_fHPD");
    index_LOOSE_AOD_N90Hits_ = index_type(&bits_, "LOOSE_AOD_N90Hits");
    index_LOOSE_AOD_EMF_     = index_type(&bits_, "LOOSE_AOD_EMF");

    index_LOOSE_fHPD_        = index_type(&bits_, "LOOSE_fHPD");
    index_LOOSE_N90Hits_     = index_type(&bits_, "LOOSE_N90Hits");
    index_LOOSE_EMF_         = index_type(&bits_, "LOOSE_EMF");

    index_TIGHT_fHPD_        = index_type(&bits_, "TIGHT_fHPD");
    index_TIGHT_EMF_         = index_type(&bits_, "TIGHT_EMF");

    index_LOOSE_nHit_        = index_type(&bits_, "LOOSE_nHit");
    index_LOOSE_als_         = index_type(&bits_, "LOOSE_als");
    index_LOOSE_fls_         = index_type(&bits_, "LOOSE_fls");
    index_LOOSE_foot_        = index_type(&bits_, "LOOSE_foot");

    index_TIGHT_nHit_        = index_type(&bits_, "TIGHT_nHit");
    index_TIGHT_als_         = index_type(&bits_, "TIGHT_als");
    index_TIGHT_fls_         = index_type(&bits_, "TIGHT_fls");
    index_TIGHT_foot_        = index_type(&bits_, "TIGHT_foot");
    index_widths_            = index_type(&bits_, "widths");
    index_EF_N90Hits_        = index_type(&bits_, "EF_N90Hits");
    index_EF_EMF_            = index_type(&bits_, "EF_EMF");

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
      const std::vector<DetId>& cellIDs = towers[iTower]->constituents();  // cell == recHit
      nHit += cellIDs.size();
    }
    return nHit;
  }

  //
  // Accessor from PAT jets
  //
  bool operator()( const pat::Jet & jet, pat::strbitset & ret )
  {
    if ( ! jet.isCaloJet() && !jet.isJPTJet() ) {
      edm::LogWarning( "NYI" )<<"Criteria for pat::Jet-s other than CaloJets and JPTJets are not yet implemented";
      return false;
    }
    if ( version_ == CRAFT08 ) return craft08Cuts( jet.p4(), jet.emEnergyFraction(), jet.jetID(), ret );
    if ( version_ == PURE09 || version_ == DQM09 ) {
      unsigned int nHit = count_hits( jet.getCaloConstituents() );
      if ( jet.currentJECLevel() == "Uncorrected" ) {
	return fwd09Cuts( jet.p4(), jet.emEnergyFraction(), jet.etaetaMoment(), jet.phiphiMoment(), nHit,
			  jet.jetID(), ret );
      }
      else {
	return fwd09Cuts( jet.correctedP4("Uncorrected"), jet.emEnergyFraction(), jet.etaetaMoment(), jet.phiphiMoment(), nHit,
			  jet.jetID(), ret );
      }
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
		   pat::strbitset & ret )
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
		   pat::strbitset & ret )
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
		    pat::strbitset & ret)
  {

    ret.set(false);

    // cache some variables
    double abs_eta = TMath::Abs( correctedP4.eta() );
    double corrPt = correctedP4.pt();
    double emf = emEnergyFraction;

    if ( ignoreCut(index_MINIMAL_EMF_) || abs_eta > 2.6 || emf > 0.01 ) passCut( ret, index_MINIMAL_EMF_);

    if ( quality_ == LOOSE_AOD ) {
      // loose fhpd cut from aod
      if ( ignoreCut(index_LOOSE_AOD_fHPD_)    || jetID.approximatefHPD < 0.98 ) passCut( ret, index_LOOSE_AOD_fHPD_);
      // loose n90 hits cut
      if ( ignoreCut(index_LOOSE_AOD_N90Hits_) || jetID.hitsInN90 > 1 ) passCut( ret, index_LOOSE_AOD_N90Hits_);

      // loose EMF Cut from aod
      bool emf_loose = true;
      if( abs_eta <= 2.6 ) { // HBHE
	if( emEnergyFraction <= 0.01 ) emf_loose = false;
      } else {                // HF
	if( emEnergyFraction <= -0.9 ) emf_loose = false;
	if( corrPt > 80 && emEnergyFraction >= 1 ) emf_loose = false;
      }
      if ( ignoreCut(index_LOOSE_AOD_EMF_) || emf_loose ) passCut(ret, index_LOOSE_AOD_EMF_);

    }
    else if ( quality_ == LOOSE || quality_ == TIGHT ) {
      // loose fhpd cut
      if ( ignoreCut(index_LOOSE_fHPD_)    || jetID.fHPD < 0.98 ) passCut( ret, index_LOOSE_fHPD_);
      // loose n90 hits cut
      if ( ignoreCut(index_LOOSE_N90Hits_) || jetID.n90Hits > 1 ) passCut( ret, index_LOOSE_N90Hits_);

      // loose EMF Cut
      bool emf_loose = true;
      if( abs_eta <= 2.6 ) { // HBHE
	if( emEnergyFraction <= 0.01 ) emf_loose = false;
      } else {                // HF
	if( emEnergyFraction <= -0.9 ) emf_loose = false;
	if( corrPt > 80 && emEnergyFraction >= 1 ) emf_loose = false;
      }
      if ( ignoreCut(index_LOOSE_EMF_) || emf_loose ) passCut(ret, index_LOOSE_EMF_);

      if ( quality_ == TIGHT ) {
	// tight fhpd cut
	bool tight_fhpd = true;
	if ( corrPt >= 25 && jetID.fHPD >= 0.95 ) tight_fhpd = false; // this was supposed to use raw pT, see AN2009/087 :-(
	if ( ignoreCut(index_TIGHT_fHPD_) || tight_fhpd ) passCut(ret, index_TIGHT_fHPD_);

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
	if ( ignoreCut(index_TIGHT_EMF_) || tight_emf ) passCut(ret, index_TIGHT_EMF_);
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
		  pat::strbitset & ret)
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

    if ( (!HBHE) || ignoreCut(index_MINIMAL_EMF_) || emf > 0.01 ) passCut( ret, index_MINIMAL_EMF_);

    // loose fhpd cut from AOD
    if ( (!HBHE) || ignoreCut(index_LOOSE_AOD_fHPD_) || jetID.approximatefHPD < 0.98 ) passCut( ret, index_LOOSE_AOD_fHPD_);
    // loose n90 hits cut from AOD
    if ( (!HBHE) || ignoreCut(index_LOOSE_AOD_N90Hits_) || jetID.hitsInN90 > 1 ) passCut( ret, index_LOOSE_AOD_N90Hits_);

    // loose fhpd cut
    if ( (!HBHE) || ignoreCut(index_LOOSE_fHPD_) || jetID.fHPD < 0.98 ) passCut( ret, index_LOOSE_fHPD_);
    // loose n90 hits cut
    if ( (!HBHE) || ignoreCut(index_LOOSE_N90Hits_) || jetID.n90Hits > 1 ) passCut( ret, index_LOOSE_N90Hits_);

    // tight fhpd cut
    if ( (!HBHE) || ignoreCut(index_TIGHT_fHPD_) || rawPt < 25 || jetID.fHPD < 0.95 ) passCut(ret, index_TIGHT_fHPD_);

    // tight emf cut
    if ( (!HBHE) || ignoreCut(index_TIGHT_EMF_) || HB || rawPt < 55 || emf < 1 ) passCut(ret, index_TIGHT_EMF_);


    // EF - these cuts are only used in "tight", but there's no need for this test here.

    if( (!EF) || ignoreCut( index_EF_N90Hits_ )
	|| jetID.n90Hits > 1 + 1.5 * TMath::Max( 0., lnpt - 1.5 ) )
      passCut( ret, index_EF_N90Hits_ );

    if( (!EF) || ignoreCut( index_EF_EMF_ )
	|| emf > TMath::Max( -0.9, -0.1 - 0.05 * TMath::Power( TMath::Max( 0., 5 - lnpt ), 2. ) ) )
      passCut( ret, index_EF_EMF_ );

    // both EF and HF

    if( ( !( EF || HF ) ) || ignoreCut( index_TIGHT_fls_ )
	|| ( EF && jetID.fLS < TMath::Min( 0.8, 0.1 + 0.016 * TMath::Power( TMath::Max( 0., 6 - lnpt ), 2.5 ) ) )
	|| ( HFa && jetID.fLS < TMath::Min( 0.6, 0.05 + 0.045 * TMath::Power( TMath::Max( 0., 7.5 - lnE ), 2.2 ) ) )
	|| ( HFb && jetID.fLS < TMath::Min( 0.1, 0.05 + 0.07 * TMath::Power( TMath::Max( 0., 7.8 - lnE ), 2. ) ) ) )
      passCut( ret, index_TIGHT_fls_ );

    if( ( !( EF || HF ) ) || ignoreCut( index_widths_ )
	|| ( 1E-10 < etaWidth && etaWidth < 0.12 &&
	     1E-10 < phiWidth && phiWidth < 0.12 ) )
      passCut( ret, index_widths_ );

    // HF cuts

    if( (!HF) || ignoreCut( index_LOOSE_nHit_ )
	|| ( HFa && nHit > 1 + 2.4*( lnpt - 1. ) )
	|| ( HFb && nHit > 1 + 3.*( lnpt - 1. ) ) )
      passCut( ret, index_LOOSE_nHit_ );

    if( (!HF) || ignoreCut( index_LOOSE_als_ )
	|| ( emf < 0.6 + 0.05 * TMath::Power( TMath::Max( 0., 9 - lnE ), 1.5 ) &&
	     emf > -0.2 - 0.041 * TMath::Power( TMath::Max( 0., 7.5 - lnE ), 2.2 ) ) )
      passCut( ret, index_LOOSE_als_ );

    if( (!HF) || ignoreCut( index_LOOSE_fls_ )
	|| ( HFa && jetID.fLS < TMath::Min( 0.9, 0.1 + 0.05 * TMath::Power( TMath::Max( 0., 7.5 - lnE ), 2.2 ) ) )
	|| ( HFb && jetID.fLS < TMath::Min( 0.6, 0.1 + 0.065 * TMath::Power( TMath::Max( 0., 7.5 - lnE ), 2.2 ) ) ) )
      passCut( ret, index_LOOSE_fls_ );

    if( (!HF) || ignoreCut( index_LOOSE_foot_ )
	|| jetID.fHFOOT < 0.9 )
      passCut( ret, index_LOOSE_foot_ );

    if( (!HF) || ignoreCut( index_TIGHT_nHit_ )
	|| ( HFa && nHit > 1 + 2.7*( lnpt - 0.8 ) )
	|| ( HFb && nHit > 1 + 3.5*( lnpt - 0.8 ) ) )
      passCut( ret, index_TIGHT_nHit_ );

    if( (!HF) || ignoreCut( index_TIGHT_als_ )
	|| ( emf < 0.5 + 0.057 * TMath::Power( TMath::Max( 0., 9 - lnE ), 1.5 ) &&
	     emf > TMath::Max( -0.6, -0.1 - 0.026 * TMath::Power( TMath::Max( 0., 8 - lnE ), 2.2 ) ) ) )
      passCut( ret, index_TIGHT_als_ );

    if( (!HF) || ignoreCut( index_TIGHT_foot_ )
	|| jetID.fLS < 0.5 )
      passCut( ret, index_TIGHT_foot_ );

    setIgnored( ret );

    return (bool)ret;
  }

 private: // member variables

  Version_t version_;
  Quality_t quality_;

  index_type index_MINIMAL_EMF_       ;

  index_type index_LOOSE_AOD_fHPD_    ;
  index_type index_LOOSE_AOD_N90Hits_ ;
  index_type index_LOOSE_AOD_EMF_     ;

  index_type index_LOOSE_fHPD_        ;
  index_type index_LOOSE_N90Hits_     ;
  index_type index_LOOSE_EMF_         ;

  index_type index_TIGHT_fHPD_        ;
  index_type index_TIGHT_EMF_         ;

  index_type index_LOOSE_nHit_        ;
  index_type index_LOOSE_als_         ;
  index_type index_LOOSE_fls_         ;
  index_type index_LOOSE_foot_        ;

  index_type index_TIGHT_nHit_        ;
  index_type index_TIGHT_als_         ;
  index_type index_TIGHT_fls_         ;
  index_type index_TIGHT_foot_        ;
  index_type index_widths_            ;
  index_type index_EF_N90Hits_        ;
  index_type index_EF_EMF_            ;

};

#endif
