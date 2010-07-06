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
  \version  $Id: PFJetIDSelectionFunctor.h,v 1.7.2.1 2010/04/27 14:56:32 srappocc Exp $
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
    push_back("NCH" );
    push_back("nConstituents");


    // Set some default cuts for LOOSE, TIGHT
    if ( quality_ == LOOSE ) {
      set("CHF", 0.0);
      set("NHF", 1.0);
      set("CEF", 1.0);
      set("NEF", 1.0);
      set("NCH", 0);
      set("nConstituents", 1);
    } else if ( quality_ == TIGHT ) {
      set("CHF", 0.0);
      set("NHF", 0.9);
      set("CEF", 1.0);
      set("NEF", 0.9);
      set("NCH", 0);
      set("nConstituents", 1);      
    }


    // Now check the configuration to see if the user changed anything
    if ( params.exists("CHF") ) set("CHF", params.getParameter<double>("CHF") );
    if ( params.exists("NHF") ) set("NHF", params.getParameter<double>("NHF") );
    if ( params.exists("CEF") ) set("CEF", params.getParameter<double>("CEF") );
    if ( params.exists("NEF") ) set("NEF", params.getParameter<double>("NEF") );
    if ( params.exists("NCH") ) set("NCH", params.getParameter<int>   ("NCH") );
    if ( params.exists("nConstuents") ) set("nConstituents", params.getParameter<int> ("nConstituents") );

    retInternal_ = getBitTemplate();
    
  }

  // 
  // Accessor from PAT jets
  // 
  bool operator()( const pat::Jet & jet, pat::strbitset & ret )  
  {
    if ( version_ == FIRSTDATA ) return firstDataCuts( jet, ret );
    else {
      return false;
    }
  }
  using Selector<pat::Jet>::operator();

  // 
  // Accessor from *CORRECTED* 4-vector, EMF, and Jet ID. 
  // This can be used with reco quantities. 
  // 
  bool operator()( reco::PFJet const & jet, 
		   pat::strbitset & ret )  
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
		      pat::strbitset & ret) 
  {    

    // cache some variables
    double chf = 0.0;
    double nhf = 0.0;
    double cef = 0.0;
    double nef = 0.0;
    int    nch = 0;
    int    nconstituents = 0;

    // Have to do this because pat::Jet inherits from reco::Jet but not reco::PFJet
    reco::PFJet const * pfJet = dynamic_cast<reco::PFJet const *>(&jet);
    pat::Jet const * patJet = dynamic_cast<pat::Jet const *>(&jet);

    if ( patJet != 0 ) {
      chf = patJet->chargedHadronEnergyFraction();
      nhf = patJet->neutralHadronEnergyFraction();
      cef = patJet->chargedEmEnergyFraction();
      nef = patJet->neutralEmEnergyFraction();
      nch = patJet->chargedMultiplicity();
      nconstituents = patJet->numberOfDaughters();
    } else if ( pfJet != 0 ) {
      chf = pfJet->chargedHadronEnergyFraction();
      nhf = pfJet->neutralHadronEnergyFraction();
      cef = pfJet->chargedEmEnergyFraction();
      nef = pfJet->neutralEmEnergyFraction();
      nch = pfJet->chargedMultiplicity();
      nconstituents = pfJet->numberOfDaughters();
    }


    // Cuts for all |eta|:
    if ( ignoreCut("nConstituents") || nconstituents > cut("nConstituents", int() ) ) passCut( ret, "nConstituents");
    if ( ignoreCut("NEF")           || ( nef < cut("NEF", double()) ) ) passCut( ret, "NEF");
    if ( ignoreCut("NHF")           || ( nhf < cut("NHF", double()) ) ) passCut( ret, "NHF");    
    // Cuts for |eta| < 2.4:
    if ( ignoreCut("CEF")           || ( cef < cut("CEF", double()) || std::abs(jet.eta()) > 2.4 ) ) passCut( ret, "CEF");
    if ( ignoreCut("CHF")           || ( chf > cut("CHF", double()) || std::abs(jet.eta()) > 2.4 ) ) passCut( ret, "CHF");
    if ( ignoreCut("NCH")           || ( nch > cut("NCH", int())    || std::abs(jet.eta()) > 2.4 ) ) passCut( ret, "NCH");    

    setIgnored( ret );
    return (bool)ret;
  }
  
 private: // member variables
  
  Version_t version_;
  Quality_t quality_;
  
};

#endif
