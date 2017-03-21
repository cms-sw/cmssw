#ifndef PhysicsTools_SelectorUtils_interface_PFJetIDSelectionFunctor_h
#define PhysicsTools_SelectorUtils_interface_PFJetIDSelectionFunctor_h

/**
  \class    PFJetIDSelectionFunctor PFJetIDSelectionFunctor.h "PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h"
  \brief    PF Jet selector for pat::Jets

  Selector functor for pat::Jets that implements quality cuts based on
  studies of noise patterns.

  Please see https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATSelectors
  for a general overview of the selectors.
*/

#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"

#include <TMath.h>
class PFJetIDSelectionFunctor : public Selector<pat::Jet>  {

 public: // interface

  enum Version_t { FIRSTDATA, RUNIISTARTUP, WINTER16, N_VERSIONS };
  enum Quality_t { LOOSE, TIGHT, N_QUALITY};

  PFJetIDSelectionFunctor() {}

#ifndef __GCCXML__
  PFJetIDSelectionFunctor( edm::ParameterSet const & params, edm::ConsumesCollector& iC ) :
    PFJetIDSelectionFunctor(params)
  {}
#endif

 PFJetIDSelectionFunctor( edm::ParameterSet const & params )
 {
   std::string versionStr = params.getParameter<std::string>("version");
   std::string qualityStr = params.getParameter<std::string>("quality");

   if ( versionStr == "FIRSTDATA" )
     version_ = FIRSTDATA;
   else if( versionStr == "RUNIISTARTUP") 
     version_ = RUNIISTARTUP;  
   // WINTER16 implements most recent (as of Feb 2017) JetID criteria
   // See: https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVRun2016
   else if( versionStr == "WINTER16") 
     version_ = WINTER16;  
   else version_ = WINTER16;//set WINTER16 as default
   

   if      ( qualityStr == "LOOSE") quality_ = LOOSE;
   else if ( qualityStr == "TIGHT") quality_ = TIGHT;
   else quality_ = LOOSE;

    push_back("CHF" );
    push_back("NHF" );
    push_back("CEF" );
    push_back("NEF" );
    push_back("NCH" );
    push_back("nConstituents");
    if(version_ == RUNIISTARTUP ){
      push_back("NEF_FW");
      push_back("nNeutrals_FW");
    }
    if(version_ == WINTER16 ){
      push_back("NHF_EC");
      push_back("NEF_EC");
      push_back("nNeutrals_EC");
      push_back("NEF_FW");
      push_back("nNeutrals_FW");
    }
 
 

    // Set some default cuts for LOOSE, TIGHT
    if ( quality_ == LOOSE ) {
      set("CHF", 0.0);
      set("NHF", 0.99);
      set("CEF", 0.99);
      set("NEF", 0.99);
      set("NCH", 0);
      set("nConstituents", 1);
      if(version_ == RUNIISTARTUP){
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }
      if(version_ == WINTER16){
	set("NHF_EC",0.98);
	set("NEF_EC",0.01);
	set("nNeutrals_EC",2);
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }


    } else if ( quality_ == TIGHT ) {
      set("CHF", 0.0);
      set("NHF", 0.9);
      set("CEF", 0.99);
      set("NEF", 0.9);
      set("NCH", 0);
      set("nConstituents", 1);
      if(version_ == RUNIISTARTUP){
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }
      if(version_ == WINTER16){
	set("NHF_EC",0.98);
	set("NEF_EC",0.01);
	set("nNeutrals_EC",2);
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }

    }


    // Now check the configuration to see if the user changed anything
    if ( params.exists("CHF") ) set("CHF", params.getParameter<double>("CHF") );
    if ( params.exists("NHF") ) set("NHF", params.getParameter<double>("NHF") );
    if ( params.exists("CEF") ) set("CEF", params.getParameter<double>("CEF") );
    if ( params.exists("NEF") ) set("NEF", params.getParameter<double>("NEF") );
    if ( params.exists("NCH") ) set("NCH", params.getParameter<int>   ("NCH") );
    if ( params.exists("nConstituents") ) set("nConstituents", params.getParameter<int> ("nConstituents") );
    if(version_ == RUNIISTARTUP){
      if ( params.exists("NEF_FW") ) set("NEF_FW", params.getParameter<double> ("NEF_FW") );
      if ( params.exists("nNeutrals_FW") ) set("nNeutrals_FW", params.getParameter<int> ("nNeutrals_FW") );
    }
    if(version_ == WINTER16){
      if ( params.exists("NHF_EC") ) set("NHF_EC", params.getParameter<int> ("NHF_EC") );
      if ( params.exists("NEF_EC") ) set("NEF_EC", params.getParameter<int> ("NEF_EC") );
      if ( params.exists("nNeutrals_EC") ) set("nNeutrals_EC", params.getParameter<int> ("nNeutrals_EC") );
      if ( params.exists("NEF_FW") ) set("NEF_FW", params.getParameter<double> ("NEF_FW") );
      if ( params.exists("nNeutrals_FW") ) set("nNeutrals_FW", params.getParameter<int> ("nNeutrals_FW") );
    }


    if ( params.exists("cutsToIgnore") )
      setIgnoredCuts( params.getParameter<std::vector<std::string> >("cutsToIgnore") );


    indexNConstituents_ = index_type (&bits_, "nConstituents");
    indexNEF_ = index_type (&bits_, "NEF");
    indexNHF_ = index_type (&bits_, "NHF");
    indexCEF_ = index_type (&bits_, "CEF");
    indexCHF_ = index_type (&bits_, "CHF");
    indexNCH_ = index_type (&bits_, "NCH");
    if(version_ == RUNIISTARTUP){
      indexNEF_FW_ = index_type (&bits_, "NEF_FW");
      indexNNeutrals_FW_ = index_type (&bits_, "nNeutrals_FW");
    }
    if(version_ == WINTER16){
      indexNHF_EC_ = index_type (&bits_, "NHF_EC");
      indexNEF_EC_ = index_type (&bits_, "NEF_EC");
      indexNNeutrals_EC_ = index_type (&bits_, "nNeutrals_EC");
      indexNEF_FW_ = index_type (&bits_, "NEF_FW");
      indexNNeutrals_FW_ = index_type (&bits_, "nNeutrals_FW");
    }

    retInternal_ = getBitTemplate();

  }


 PFJetIDSelectionFunctor( Version_t version,
			  Quality_t quality ) :
  version_(version), quality_(quality)
 {

    push_back("CHF" );
    push_back("NHF" );
    push_back("CEF" );
    push_back("NEF" );
    push_back("NCH" );
    push_back("nConstituents");
    if(version_ == RUNIISTARTUP){
      push_back("NEF_FW");
      push_back("nNeutrals_FW");
    }
    if(version_ == WINTER16){
      push_back("NHF_EC");
      push_back("NEF_EC");
      push_back("nNeutrals_EC");
      push_back("NEF_FW");
      push_back("nNeutrals_FW");
    }


    // Set some default cuts for LOOSE, TIGHT
    if ( quality_ == LOOSE ) {
      set("CHF", 0.0);
      set("NHF", 0.99);
      set("CEF", 0.99);
      set("NEF", 0.99);
      set("NCH", 0);
      set("nConstituents", 1);
      if(version_ == RUNIISTARTUP){
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }
      if(version_ == WINTER16){
	set("NHF_EC",0.98);
	set("NEF_EC",0.01);
	set("nNeutrals_EC",2);
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }

    } else if ( quality_ == TIGHT ) {
      set("CHF", 0.0);
      set("NHF", 0.9);
      set("CEF", 0.99);
      set("NEF", 0.9);
      set("NCH", 0);
      set("nConstituents", 1);
      if(version_ == RUNIISTARTUP){
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }
      if(version_ == WINTER16){
	set("NHF_EC",0.98);
	set("NEF_EC",0.01);
	set("nNeutrals_EC",2);
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }
    }


    indexNConstituents_ = index_type (&bits_, "nConstituents");
    indexNEF_ = index_type (&bits_, "NEF");
    indexNHF_ = index_type (&bits_, "NHF");
    indexCEF_ = index_type (&bits_, "CEF");
    indexCHF_ = index_type (&bits_, "CHF");
    indexNCH_ = index_type (&bits_, "NCH");
    if(version_ == RUNIISTARTUP){
      indexNEF_FW_ = index_type (&bits_, "NEF_FW");
      indexNNeutrals_FW_ = index_type (&bits_, "nNeutrals_FW");
    }
    if(version_ == WINTER16){
      indexNHF_EC_ = index_type (&bits_, "NHF_EC");
      indexNEF_EC_ = index_type (&bits_, "NEF_EC");
      indexNNeutrals_EC_ = index_type (&bits_, "nNeutrals_EC");
      indexNEF_FW_ = index_type (&bits_, "NEF_FW");
      indexNNeutrals_FW_ = index_type (&bits_, "nNeutrals_FW");
    }


    retInternal_ = getBitTemplate();
 }


  //
  // Accessor from PAT jets
  //
  bool operator()( const pat::Jet & jet, pat::strbitset & ret )
  {
    if ( version_ == FIRSTDATA || version_ == RUNIISTARTUP || version_ == WINTER16) {
      if ( jet.currentJECLevel() == "Uncorrected" || !jet.jecSetsAvailable() )
	return firstDataCuts( jet, ret, version_);
      else
	return firstDataCuts( jet.correctedJet("Uncorrected"), ret, version_ );
    }
    else {
      return false;
    }
  }
  using Selector<pat::Jet>::operator();

  //
  // Accessor from *CORRECTED* 4-vector, EMF, and Jet ID.
  // This can be used with reco quantities.
  //
  bool operator()( const reco::PFJet & jet, pat::strbitset & ret )
  {
    if ( version_ == FIRSTDATA || version_ == RUNIISTARTUP || version_ == WINTER16  ){ return firstDataCuts( jet, ret, version_);
    }
    else {
      return false;
    }
  }

  bool operator()( const reco::PFJet & jet )
  {
    retInternal_.set(false);
    operator()(jet, retInternal_);
    setIgnored(retInternal_);
    return (bool)retInternal_;
  }

  //
  // cuts based on craft 08 analysis.
  //
  bool firstDataCuts( reco::Jet const & jet,
		      pat::strbitset & ret, Version_t version_)
  {
    ret.set(false);

    // cache some variables
    double chf = 0.0;
    double nhf = 0.0;
    double cef = 0.0;
    double nef = 0.0;
    int    nch = 0;
    int    nconstituents = 0;
    int    nneutrals = 0;

    // Have to do this because pat::Jet inherits from reco::Jet but not reco::PFJet
    reco::PFJet const * pfJet = dynamic_cast<reco::PFJet const *>(&jet);
    pat::Jet const * patJet = dynamic_cast<pat::Jet const *>(&jet);
    reco::BasicJet const * basicJet = dynamic_cast<reco::BasicJet const *>(&jet);

    if ( patJet != 0 ) {
      if ( patJet->isPFJet() ) {
	chf = patJet->chargedHadronEnergyFraction();
	nhf = patJet->neutralHadronEnergyFraction();
	cef = patJet->chargedEmEnergyFraction();
	nef = patJet->neutralEmEnergyFraction();
	nch = patJet->chargedMultiplicity();
	nconstituents = patJet->numberOfDaughters();
	nneutrals = patJet->neutralMultiplicity();
      }
      // Handle the special case where this is a composed jet for
      // subjet analyses
      else if ( patJet->isBasicJet() ) {
	double e_chf = 0.0;
	double e_nhf = 0.0;
	double e_cef = 0.0;
	double e_nef = 0.0;
	nch = 0;
	nconstituents = 0;
	nneutrals = 0;

	for ( reco::Jet::const_iterator ibegin = patJet->begin(),
		iend = patJet->end(), isub = ibegin;
	      isub != iend; ++isub ) {
	  reco::PFJet const * pfsub = dynamic_cast<reco::PFJet const *>( &*isub );
	  e_chf += pfsub->chargedHadronEnergy();
	  e_nhf += pfsub->neutralHadronEnergy();
	  e_cef += pfsub->chargedEmEnergy();
	  e_nef += pfsub->neutralEmEnergy();
	  nch += pfsub->chargedMultiplicity();
	  nconstituents += pfsub->numberOfDaughters();
	  nneutrals += pfsub->neutralMultiplicity();
	}
	double e = patJet->energy();
	if ( e > 0.000001 ) {
	  chf = e_chf / e;
	  nhf = e_nhf / e;
	  cef = e_cef / e;
	  nef = e_nef / e;
	} else {
	  chf = nhf = cef = nef = 0.0;
	}
      }
    } // end if pat jet
    else if ( pfJet != 0 ) {
      // CV: need to compute energy fractions in a way that works for corrected as well as for uncorrected PFJets
      double jetEnergyUncorrected =
	pfJet->chargedHadronEnergy()
       + pfJet->neutralHadronEnergy()
       + pfJet->photonEnergy()
       + pfJet->electronEnergy()
       + pfJet->muonEnergy()
       + pfJet->HFEMEnergy();
      if ( jetEnergyUncorrected > 0. ) {
	chf = pfJet->chargedHadronEnergy() / jetEnergyUncorrected;
        nhf = pfJet->neutralHadronEnergy() / jetEnergyUncorrected;
        cef = pfJet->chargedEmEnergy() / jetEnergyUncorrected;
        nef = pfJet->neutralEmEnergy() / jetEnergyUncorrected;
      }
      nch = pfJet->chargedMultiplicity();
      nconstituents = pfJet->numberOfDaughters();
      nneutrals = pfJet->neutralMultiplicity();
    } // end if PF jet
    // Handle the special case where this is a composed jet for
    // subjet analyses
    else if ( basicJet != 0 ) {
      double e_chf = 0.0;
      double e_nhf = 0.0;
      double e_cef = 0.0;
      double e_nef = 0.0;
      nch = 0;
      nconstituents = 0;
      for ( reco::Jet::const_iterator ibegin = basicJet->begin(),
	      iend = patJet->end(), isub = ibegin;
	    isub != iend; ++isub ) {
	reco::PFJet const * pfsub = dynamic_cast<reco::PFJet const *>( &*isub );
	e_chf += pfsub->chargedHadronEnergy();
	e_nhf += pfsub->neutralHadronEnergy();
	e_cef += pfsub->chargedEmEnergy();
	e_nef += pfsub->neutralEmEnergy();
	nch += pfsub->chargedMultiplicity();
	nconstituents += pfsub->numberOfDaughters();
	nneutrals += pfsub->neutralMultiplicity();
      }
      double e = basicJet->energy();
      if ( e > 0.000001 ) {
	chf = e_chf / e;
	nhf = e_nhf / e;
	cef = e_cef / e;
	nef = e_nef / e;
      }
    } // end if basic jet



   // Cuts for |eta| < 2.4 for FIRSTDATA, RUNIISTARTUP and WINTER16
    if ( ignoreCut(indexCEF_)           || ( cef < cut(indexCEF_, double()) || std::abs(jet.eta()) > 2.4 ) ) passCut( ret, indexCEF_);
    if ( ignoreCut(indexCHF_)           || ( chf > cut(indexCHF_, double()) || std::abs(jet.eta()) > 2.4 ) ) passCut( ret, indexCHF_);
    if ( ignoreCut(indexNCH_)           || ( nch > cut(indexNCH_, int())    || std::abs(jet.eta()) > 2.4 ) ) passCut( ret, indexNCH_);

    if(version_ == FIRSTDATA){// Cuts for all eta for FIRSTDATA
      if ( ignoreCut(indexNConstituents_) || ( nconstituents > cut(indexNConstituents_, int()) ) ) passCut( ret, indexNConstituents_);
      if ( ignoreCut(indexNEF_)           || ( nef < cut(indexNEF_, double()) ) ) passCut( ret, indexNEF_);
      if ( ignoreCut(indexNHF_)           || ( nhf < cut(indexNHF_, double()) ) ) passCut( ret, indexNHF_);
    }else if(version_ == RUNIISTARTUP){
      // Cuts for |eta| <= 3.0 for RUNIISTARTUP scenario
      if ( ignoreCut(indexNConstituents_) || ( nconstituents > cut(indexNConstituents_, int()) || std::abs(jet.eta()) > 3.0 ) ) passCut( ret, indexNConstituents_);
      if ( ignoreCut(indexNEF_)           || ( nef < cut(indexNEF_, double())  || std::abs(jet.eta()) > 3.0 ) ) passCut( ret, indexNEF_);
      if ( ignoreCut(indexNHF_)           || ( nhf < cut(indexNHF_, double())  || std::abs(jet.eta()) > 3.0 ) ) passCut( ret, indexNHF_);
      // Cuts for |eta| > 3.0 for RUNIISTARTUP scenario
      if ( ignoreCut(indexNEF_FW_)           || ( nef < cut(indexNEF_FW_, double()) || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNEF_FW_);
      if ( ignoreCut(indexNNeutrals_FW_) || ( nneutrals > cut(indexNNeutrals_FW_, int())    || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNNeutrals_FW_);
    }
    else if(version_ == WINTER16){
      // Cuts for |eta| <= 2.7 for WINTER16 scenario
      if ( ignoreCut(indexNConstituents_) || ( nconstituents > cut(indexNConstituents_, int()) || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexNConstituents_);
      if ( ignoreCut(indexNEF_)           || ( nef < cut(indexNEF_, double())  || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexNEF_);
      if ( ignoreCut(indexNHF_)           || ( nhf < cut(indexNHF_, double())  || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexNHF_);

      // Cuts for 2.7 < |eta| <= 3.0 for WINTER16 scenario
      if ( ignoreCut(indexNHF_EC_)        || ( nhf < cut(indexNHF_EC_, double())  || std::abs(jet.eta()) <= 2.7 || std::abs(jet.eta()) > 3.0)  ) passCut( ret, indexNHF_EC_);
      if ( ignoreCut(indexNEF_EC_)        || ( nef > cut(indexNEF_EC_, double())  || std::abs(jet.eta()) <= 2.7 || std::abs(jet.eta()) > 3.0)  ) passCut( ret, indexNEF_EC_);
      if ( ignoreCut(indexNNeutrals_EC_)  || ( nneutrals > cut(indexNNeutrals_EC_, int())  || std::abs(jet.eta()) <= 2.7 || std::abs(jet.eta()) > 3.0)  ) passCut( ret, indexNNeutrals_EC_);

      // Cuts for |eta| > 3.0 for WINTER16 scenario
      if ( ignoreCut(indexNEF_FW_)           || ( nef < cut(indexNEF_FW_, double()) || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNEF_FW_);
      if ( ignoreCut(indexNNeutrals_FW_) || ( nneutrals > cut(indexNNeutrals_FW_, int())    || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNNeutrals_FW_);
    }


    //std::cout << "<PFJetIDSelectionFunctor::firstDataCuts>:" << std::endl;
    //std::cout << " jet: Pt = " << jet.pt() << ", eta = " << jet.eta() << ", phi = " << jet.phi() << std::endl;
    //ret.print(std::cout);

    setIgnored( ret );
    return (bool)ret;
  }

 private: // member variables

  Version_t version_;
  Quality_t quality_;

  index_type indexNConstituents_;
  index_type indexNEF_;
  index_type indexNHF_;
  index_type indexCEF_;
  index_type indexCHF_;
  index_type indexNCH_;

  index_type indexNEF_FW_;
  index_type indexNNeutrals_FW_;

  index_type indexNHF_EC_;
  index_type indexNEF_EC_;
  index_type indexNNeutrals_EC_;


};

#endif
