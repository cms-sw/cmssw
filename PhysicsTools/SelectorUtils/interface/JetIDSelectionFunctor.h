#ifndef PhysicsTools_PatUtils_interface_JetIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_JetIDSelectionFunctor_h


/**
  \class    JetIDSelectionFunctor JetIDSelectionFunctor.h "PhysicsTools/Utilities/interface/JetIDSelectionFunctor.h"
  \brief    Jet selector for pat::Jets

  Selector functor for pat::Jets that implements quality cuts based on
  studies of noise patterns. 

  Please see https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATSelectors
  for a general overview of the selectors. 

  \author Salvatore Rappoccio
  \version  $Id: JetIDSelectionFunctor.h,v 1.5 2010/02/10 20:06:25 srappocc Exp $
*/




#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TMath.h>
class JetIDSelectionFunctor : public Selector<pat::Jet>  {

 public: // interface

  enum Version_t { CRAFT08, N_VERSIONS };
  enum Quality_t { MINIMAL, LOOSE_AOD, LOOSE, TIGHT, N_QUALITY};
  

  JetIDSelectionFunctor( edm::ParameterSet const & parameters ) {
    std::string versionStr = parameters.getParameter<std::string>("version");
    std::string qualityStr = parameters.getParameter<std::string>("quality");
    Quality_t quality = N_QUALITY;

    if ( versionStr == "CRAFT08" ) {
      if      ( qualityStr == "MINIMAL" )   quality = MINIMAL;
      else if ( qualityStr == "LOOSE_AOD" ) quality = LOOSE_AOD;
      else if ( qualityStr == "LOOSE" )     quality = LOOSE;
      else                                  quality = TIGHT;
      
      initialize( CRAFT08, quality );
    } else {
      throw cms::Exception("InvalidInput") << "Expect version to be one of SUMMER08, FIRSTDATA," << std::endl;
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

    // all on by default
    set("MINIMAL_EMF");
    set("LOOSE_AOD_fHPD");
    set("LOOSE_AOD_N90Hits");
    set("LOOSE_AOD_EMF");
    set("LOOSE_fHPD");
    set("LOOSE_N90Hits");
    set("LOOSE_EMF");
    set("TIGHT_fHPD");
    set("TIGHT_EMF");

    // now set the return values for the ignored parts
    if ( quality_ == MINIMAL ) {
      set("LOOSE_fHPD", false );
      set("LOOSE_N90Hits", false );
      set("LOOSE_EMF", false );
      set("LOOSE_AOD_fHPD", false );
      set("LOOSE_AOD_N90Hits", false );
      set("LOOSE_AOD_EMF", false );
      set("TIGHT_fHPD", false );
      set("TIGHT_EMF", false );      
    } 
    if ( quality_ == LOOSE ) {
      set("LOOSE_AOD_fHPD", false );
      set("LOOSE_AOD_N90Hits", false );
      set("LOOSE_AOD_EMF", false );
      set("TIGHT_fHPD", false );
      set("TIGHT_EMF", false );      
    } 
    if ( quality_ == LOOSE_AOD ) {
      set("LOOSE_fHPD", false );
      set("LOOSE_N90Hits", false );
      set("LOOSE_EMF", false );
      set("TIGHT_fHPD", false );
      set("TIGHT_EMF", false );      
    }
    if ( quality_ == TIGHT ) {
      set("LOOSE_AOD_fHPD", false );
      set("LOOSE_AOD_N90Hits", false );
      set("LOOSE_AOD_EMF", false );
    }
  
    retInternal_ = getBitTemplate();
  }

  // 
  // Accessor from PAT jets
  // 
  bool operator()( const pat::Jet & jet, std::strbitset & ret )  
  {
    if ( version_ == CRAFT08 ) return craft08Cuts( jet.p4(), jet.emEnergyFraction(), jet.jetID(), ret );
    else {
      return false;
    }
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
    else {
      return false;
    }
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
  // Accessor from *CORRECTED* CaloJet and Jet ID. 
  // This can be used with reco quantities. 
  // 
  bool operator()( reco::CaloJet const & jet,
		   reco::JetID const & jetID,
		   std::strbitset & ret )  
  {
    if ( version_ == CRAFT08 ) return craft08Cuts( jet.p4(), jet.emEnergyFraction(), jetID, ret );
    else {
      return false;
    }
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
	if ( corrPt >= 25 && jetID.fHPD >= 0.95 ) tight_fhpd = false;
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
  
 private: // member variables
  
  Version_t version_;
  Quality_t quality_;
  
};

#endif
