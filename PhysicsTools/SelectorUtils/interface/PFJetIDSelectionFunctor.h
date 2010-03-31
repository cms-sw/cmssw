#ifndef PhysicsTools_PatUtils_interface_PFJetIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_PFJetIDSelectionFunctor_h


/**
  \class    PFJetIDSelectionFunctor PFJetIDSelectionFunctor.h "PhysicsTools/Utilities/interface/PFJetIDSelectionFunctor.h"
  \brief    PF Jet selector for pat::Jets

  Selector functor for pat::Jets that implements quality cuts based on
  studies of noise patterns. 

  Please see https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATSelectors
  for a general overview of the selectors. 

  \author Salvatore Rappoccio
  \version  $Id: PFJetIDSelectionFunctor.h,v 1.3 2010/03/25 03:35:21 srappocc Exp $
*/




#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"

#include <TMath.h>
class PFJetIDSelectionFunctor : public Selector<pat::Jet>  {

 public: // interface

  enum Version_t { FIRSTDATA, N_VERSIONS };
  enum Quality_t { LOOSE, TIGHT, N_QUALITY};
  
 PFJetIDSelectionFunctor( edm::ParameterSet const & params ) 
 {

   std::string versionStr = params.getParameter<std::string>("version");
   std::string qualityStr = params.getParameter<std::string>("quality");

   if ( versionStr == "FIRSTDATA" ) 
     version_ = FIRSTDATA;
   else
     version_ = FIRSTDATA;  /// will have other options eventually, most likely

   if      ( qualityStr == "LOOSE") quality_ = LOOSE;
   else if ( qualityStr == "TIGHT") quality_ = TIGHT;
   else quality_ = LOOSE;

    push_back("CHF" );
    push_back("NHF" );
    push_back("CEF" );
    push_back("NEF" );

    // all on by default
    set("CHF", params.getParameter<double>("CHF") );
    set("NHF", params.getParameter<double>("NHF") );
    set("CEF", params.getParameter<double>("CEF") );
    set("NEF", params.getParameter<double>("NEF") );

    if ( quality_ == LOOSE ) {
      set("NHF", false );
      set("NEF", false );
      set("CEF", false );
    }
    
  }

  // 
  // Accessor from PAT jets
  // 
  bool operator()( const pat::Jet & jet, std::strbitset & ret )  
  {
    if ( version_ == FIRSTDATA ) return firstDataCuts( jet, ret );
    else {
      return false;
    }
  }

  // 
  // Accessor from *CORRECTED* 4-vector, EMF, and Jet ID. 
  // This can be used with reco quantities. 
  // 
  bool operator()( reco::PFJet const & jet, 
		   std::strbitset & ret )  
  {
    if ( version_ == FIRSTDATA ) return firstDataCuts( jet, ret );
    else {
      return false;
    }
  }
  
  // 
  // cuts based on craft 08 analysis. 
  // 
  bool firstDataCuts( reco::Jet const & jet,
		      std::strbitset & ret) 
  {    

    // cache some variables
    double chf = 0.0;
    double nhf = 0.0;
    double cef = 0.0;
    double nef = 0.0;

    // Have to do this because pat::Jet inherits from reco::Jet but not reco::PFJet
    reco::PFJet const * pfJet = dynamic_cast<reco::PFJet const *>(&jet);
    pat::Jet const * patJet = dynamic_cast<pat::Jet const *>(&jet);

    if ( patJet != 0 ) {
      chf = patJet->chargedHadronEnergyFraction();
      nhf = patJet->neutralHadronEnergyFraction();
      cef = patJet->chargedEmEnergyFraction();
      nef = patJet->neutralEmEnergyFraction();
    } else if ( pfJet != 0 ) {
      chf = pfJet->chargedHadronEnergyFraction();
      nhf = pfJet->neutralHadronEnergyFraction();
      cef = pfJet->chargedEmEnergyFraction();
      nef = pfJet->neutralEmEnergyFraction();
    }

    if ( ignoreCut("CHF") || (std::abs(jet.eta()) <= 2.4 && chf > cut("CHF", double())) || (std::abs(jet.eta()) > 2.4) ) {
      passCut( ret, "CHF");

      if ( ignoreCut("NHF") || nhf < cut("NHF", double()) ) {
	passCut( ret, "NHF");
      }

      if ( ignoreCut("CEF") || cef < cut("CEF", double()) ) {
	passCut( ret, "CEF");
      }

      if ( ignoreCut("NEF") || nef < cut("NEF", double()) ) {
	passCut( ret, "NEF");
      }
    }

    setIgnored( ret );
    return (bool)ret;
  }
  
 private: // member variables
  
  Version_t version_;
  Quality_t quality_;
  
};

#endif
