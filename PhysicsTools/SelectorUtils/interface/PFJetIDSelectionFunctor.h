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

  enum Version_t { FIRSTDATA, RUNIISTARTUP, WINTER16, WINTER17, WINTER17PUPPI, N_VERSIONS };
  enum Quality_t { LOOSE, TIGHT, TIGHTLEPVETO, N_QUALITY};

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
   else if( versionStr == "WINTER17") 
     version_ = WINTER17;   
   else if( versionStr == "WINTER17PUPPI")
     version_ = WINTER17PUPPI;
   else version_ = WINTER17;//set WINTER17 as default
   

   if      ( qualityStr == "LOOSE") quality_ = LOOSE;
   else if ( qualityStr == "TIGHT") quality_ = TIGHT;
   else if ( qualityStr == "TIGHTLEPVETO") quality_ = TIGHTLEPVETO;
   else quality_ = TIGHT;

   initCuts();

    // Now check the configuration to see if the user changed anything
    if ( params.exists("CHF") ) set("CHF", params.getParameter<double>("CHF") );
    if ( params.exists("NHF") ) set("NHF", params.getParameter<double>("NHF") );
    if ((version_ != WINTER17 && version_ != WINTER17PUPPI) || quality_ != TIGHT ) {if ( params.exists("CEF") ) set("CEF", params.getParameter<double>("CEF") );}
    if ( params.exists("NEF") ) set("NEF", params.getParameter<double>("NEF") );
    if ( params.exists("NCH") ) set("NCH", params.getParameter<int>   ("NCH") );
    if ( params.exists("nConstituents") ) set("nConstituents", params.getParameter<int> ("nConstituents") );
    if(version_ == RUNIISTARTUP){
      if ( params.exists("NEF_FW") ) set("NEF_FW", params.getParameter<double> ("NEF_FW") );
      if ( params.exists("nNeutrals_FW") ) set("nNeutrals_FW", params.getParameter<int> ("nNeutrals_FW") );
    }
    if(version_ == WINTER16){
      if ( params.exists("NHF_EC") ) set("NHF_EC", params.getParameter<double> ("NHF_EC") );
      if ( params.exists("NEF_EC") ) set("NEF_EC", params.getParameter<double> ("NEF_EC") );
      if ( params.exists("nNeutrals_EC") ) set("nNeutrals_EC", params.getParameter<int> ("nNeutrals_EC") );
      if ( params.exists("NEF_FW") ) set("NEF_FW", params.getParameter<double> ("NEF_FW") );
      if ( params.exists("nNeutrals_FW") ) set("nNeutrals_FW", params.getParameter<int> ("nNeutrals_FW") );
      if ( quality_ == TIGHTLEPVETO ) {if ( params.exists("MUF") ) set("MUF", params.getParameter<double> ("MUF") );}
    }
    if(version_ == WINTER17){
      if ( params.exists("NEF_EC_L") ) set("NEF_EC_L", params.getParameter<double> ("NEF_EC_L") );
      if ( params.exists("NEF_EC_U") ) set("NEF_EC_U", params.getParameter<double> ("NEF_EC_U") );
      if ( params.exists("nNeutrals_EC") ) set("nNeutrals_EC", params.getParameter<int> ("nNeutrals_EC") );
      if ( params.exists("NHF_FW") ) set("NHF_FW", params.getParameter<double> ("NHF_FW") );
      if ( params.exists("NEF_FW") ) set("NEF_FW", params.getParameter<double> ("NEF_FW") );
      if ( params.exists("nNeutrals_FW") ) set("nNeutrals_FW", params.getParameter<int> ("nNeutrals_FW") );
      if ( quality_ == TIGHTLEPVETO ) {if ( params.exists("MUF") ) set("MUF", params.getParameter<double> ("MUF") );}
    }
    if(version_ == WINTER17PUPPI){
      if ( params.exists("NHF_EC") ) set("NHF_EC", params.getParameter<double> ("NHF_EC") );
      if ( params.exists("NHF_FW") ) set("NHF_FW", params.getParameter<double> ("NHF_FW") );
      if ( params.exists("NEF_FW") ) set("NEF_FW", params.getParameter<double> ("NEF_FW") );
      if ( params.exists("nNeutrals_FW_L") ) set("nNeutrals_FW_L", params.getParameter<int> ("nNeutrals_FW_L") );
      if ( params.exists("nNeutrals_FW_U") ) set("nNeutrals_FW_U", params.getParameter<int> ("nNeutrals_FW_U") );
      if ( quality_ == TIGHTLEPVETO ) {if ( params.exists("MUF") ) set("MUF", params.getParameter<double> ("MUF") );}
    }


    if ( params.exists("cutsToIgnore") )
      setIgnoredCuts( params.getParameter<std::vector<std::string> >("cutsToIgnore") );

    initIndex();

  }


 PFJetIDSelectionFunctor( Version_t version,
			  Quality_t quality ) :
  version_(version), quality_(quality)
 {
   initCuts();
   initIndex();
 }


  //
  // Accessor from PAT jets
  //
  bool operator()( const pat::Jet & jet, pat::strbitset & ret ) override
  {
    if ( version_ == FIRSTDATA || version_ == RUNIISTARTUP || version_ == WINTER16 || version_ == WINTER17 || version_ == WINTER17PUPPI) {
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
    if ( version_ == FIRSTDATA || version_ == RUNIISTARTUP || version_ == WINTER16 || version_ == WINTER17 || version_ == WINTER17PUPPI  ){ return firstDataCuts( jet, ret, version_);
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
    double muf = 0.0;
    int    nch = 0;
    int    nconstituents = 0;
    int    nneutrals = 0;

    // Have to do this because pat::Jet inherits from reco::Jet but not reco::PFJet
    reco::PFJet const * pfJet = dynamic_cast<reco::PFJet const *>(&jet);
    pat::Jet const * patJet = dynamic_cast<pat::Jet const *>(&jet);
    reco::BasicJet const * basicJet = dynamic_cast<reco::BasicJet const *>(&jet);

    if ( patJet != nullptr ) {
      if ( patJet->isPFJet() ) {
	chf = patJet->chargedHadronEnergyFraction();
	nhf = patJet->neutralHadronEnergyFraction();
	cef = patJet->chargedEmEnergyFraction();
	nef = patJet->neutralEmEnergyFraction();
	nch = patJet->chargedMultiplicity();
	nconstituents = patJet->numberOfDaughters();
	nneutrals = patJet->neutralMultiplicity();
        // Handle the special case of PUPPI jets with weighted multiplicities
        if (patJet->hasUserFloat("patPuppiJetSpecificProducer:puppiMultiplicity"))
           nconstituents = patJet->userFloat("patPuppiJetSpecificProducer:puppiMultiplicity");
        if (patJet->hasUserFloat("patPuppiJetSpecificProducer:neutralPuppiMultiplicity"))
           nneutrals = patJet->userFloat("patPuppiJetSpecificProducer:neutralPuppiMultiplicity");
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
	  pat::Jet const * patsub = dynamic_cast<pat::Jet const *>( &*isub );
	  if ( patsub ) {
	    e_chf += patsub->chargedHadronEnergy();
	    e_nhf += patsub->neutralHadronEnergy();
	    e_cef += patsub->chargedEmEnergy();
	    e_nef += patsub->neutralEmEnergy();
	    nch += patsub->chargedMultiplicity();
            nconstituents += patsub->numberOfDaughters();
	    nneutrals += patsub->neutralMultiplicity();
	  } else if ( pfsub ) {
            e_chf += pfsub->chargedHadronEnergy();
	    e_nhf += pfsub->neutralHadronEnergy();
	    e_cef += pfsub->chargedEmEnergy();
	    e_nef += pfsub->neutralEmEnergy();
	    nch += pfsub->chargedMultiplicity();
            nconstituents += pfsub->numberOfDaughters();
	    nneutrals += pfsub->neutralMultiplicity();
	  } else assert(0);
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
    else if ( pfJet != nullptr ) {
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
        muf = pfJet->muonEnergy() / jetEnergyUncorrected;
      }
      nch = pfJet->chargedMultiplicity();
      nconstituents = pfJet->numberOfDaughters();
      nneutrals = pfJet->neutralMultiplicity();
    } // end if PF jet
    // Handle the special case where this is a composed jet for
    // subjet analyses
    else if ( basicJet != nullptr ) {
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



   // Cuts for |eta| < 2.4 for FIRSTDATA, RUNIISTARTUP, WINTER16 and WINTER17
    if((version_ != WINTER17 && version_ != WINTER17PUPPI) ||  quality_ != TIGHT ) {if ( ignoreCut(indexCEF_)           || ( cef < cut(indexCEF_, double()) || std::abs(jet.eta()) > 2.4 ) ) passCut( ret, indexCEF_);}

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
      if ( quality_ == TIGHTLEPVETO ) {if ( ignoreCut(indexMUF_)           || ( muf < cut(indexMUF_, double()) || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexMUF_);}

      // Cuts for 2.7 < |eta| <= 3.0 for WINTER16 scenario
      if ( ignoreCut(indexNHF_EC_)        || ( nhf < cut(indexNHF_EC_, double())  || std::abs(jet.eta()) <= 2.7 || std::abs(jet.eta()) > 3.0)  ) passCut( ret, indexNHF_EC_);
      if ( ignoreCut(indexNEF_EC_)        || ( nef > cut(indexNEF_EC_, double())  || std::abs(jet.eta()) <= 2.7 || std::abs(jet.eta()) > 3.0)  ) passCut( ret, indexNEF_EC_);
      if ( ignoreCut(indexNNeutrals_EC_)  || ( nneutrals > cut(indexNNeutrals_EC_, int())  || std::abs(jet.eta()) <= 2.7 || std::abs(jet.eta()) > 3.0)  ) passCut( ret, indexNNeutrals_EC_);

      // Cuts for |eta| > 3.0 for WINTER16 scenario
      if ( ignoreCut(indexNEF_FW_)           || ( nef < cut(indexNEF_FW_, double()) || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNEF_FW_);
      if ( ignoreCut(indexNNeutrals_FW_) || ( nneutrals > cut(indexNNeutrals_FW_, int())    || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNNeutrals_FW_);
    }
    else if(version_ == WINTER17){
      // Cuts for |eta| <= 2.7 for WINTER17 scenario
      if ( ignoreCut(indexNConstituents_) || ( nconstituents > cut(indexNConstituents_, int()) || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexNConstituents_);
      if ( ignoreCut(indexNEF_)           || ( nef < cut(indexNEF_, double())  || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexNEF_);
      if ( ignoreCut(indexNHF_)           || ( nhf < cut(indexNHF_, double())  || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexNHF_);
      if ( quality_ == TIGHTLEPVETO ) {if ( ignoreCut(indexMUF_)           || ( muf < cut(indexMUF_, double()) || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexMUF_);}

      // Cuts for 2.7 < |eta| <= 3.0 for WINTER17 scenario

      if ( ignoreCut(indexNEF_EC_L_)        || ( nef > cut(indexNEF_EC_L_, double())  || std::abs(jet.eta()) <= 2.7 || std::abs(jet.eta()) > 3.0)  ) passCut( ret, indexNEF_EC_L_);
      if ( ignoreCut(indexNEF_EC_U_)        || ( nef < cut(indexNEF_EC_U_, double())  || std::abs(jet.eta()) <= 2.7 || std::abs(jet.eta()) > 3.0)  ) passCut( ret, indexNEF_EC_U_);  
      if ( ignoreCut(indexNNeutrals_EC_)  || ( nneutrals > cut(indexNNeutrals_EC_, int())  || std::abs(jet.eta()) <= 2.7 || std::abs(jet.eta()) > 3.0)  ) passCut( ret, indexNNeutrals_EC_);

      // Cuts for |eta| > 3.0 for WINTER17 scenario
      if ( ignoreCut(indexNHF_FW_)           || ( nhf > cut(indexNHF_FW_, double()) || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNHF_FW_);
      if ( ignoreCut(indexNEF_FW_)           || ( nef < cut(indexNEF_FW_, double()) || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNEF_FW_);
      if ( ignoreCut(indexNNeutrals_FW_) || ( nneutrals > cut(indexNNeutrals_FW_, int())    || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNNeutrals_FW_);

    }
    else if(version_ == WINTER17PUPPI){
      // Cuts for |eta| <= 2.7 for WINTER17 scenario
      if ( ignoreCut(indexNConstituents_) || ( nconstituents > cut(indexNConstituents_, int()) || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexNConstituents_);
      if ( ignoreCut(indexNEF_)           || ( nef < cut(indexNEF_, double())  || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexNEF_);
      if ( ignoreCut(indexNHF_)           || ( nhf < cut(indexNHF_, double())  || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexNHF_);
      if ( quality_ == TIGHTLEPVETO ) {if ( ignoreCut(indexMUF_)           || ( muf < cut(indexMUF_, double()) || std::abs(jet.eta()) > 2.7 ) ) passCut( ret, indexMUF_);}

      // Cuts for 2.7 < |eta| <= 3.0 for WINTER17 scenario

      if ( ignoreCut(indexNHF_EC_)        || ( nhf < cut(indexNHF_EC_, double())  || std::abs(jet.eta()) <= 2.7 || std::abs(jet.eta()) > 3.0)  ) passCut( ret, indexNHF_EC_);

      // Cuts for |eta| > 3.0 for WINTER17 scenario
      if ( ignoreCut(indexNHF_FW_)           || ( nhf > cut(indexNHF_FW_, double()) || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNHF_FW_);
      if ( ignoreCut(indexNEF_FW_)           || ( nef < cut(indexNEF_FW_, double()) || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNEF_FW_);
      if ( ignoreCut(indexNNeutrals_FW_L_) || ( nneutrals > cut(indexNNeutrals_FW_L_, int())    || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNNeutrals_FW_L_);
      if ( ignoreCut(indexNNeutrals_FW_U_) || ( nneutrals < cut(indexNNeutrals_FW_U_, int())    || std::abs(jet.eta()) <= 3.0 ) ) passCut( ret, indexNNeutrals_FW_U_);

    }


    //std::cout << "<PFJetIDSelectionFunctor::firstDataCuts>:" << std::endl;
    //std::cout << " jet: Pt = " << jet.pt() << ", eta = " << jet.eta() << ", phi = " << jet.phi() << std::endl;
    //ret.print(std::cout);

    setIgnored( ret );
    return (bool)ret;
  }

 private: // member variables

  void initCuts()
  {
    push_back("CHF" );
    push_back("NHF" );
    if( (version_ != WINTER17 && version_!=WINTER17PUPPI) || quality_ != TIGHT ) push_back("CEF" );
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
      if (quality_ == TIGHTLEPVETO) push_back("MUF");
    }
    if(version_ == WINTER17 ){
      push_back("NEF_EC_L");
      push_back("NEF_EC_U");
      push_back("nNeutrals_EC");
      push_back("NEF_FW");
      push_back("NHF_FW");
      push_back("nNeutrals_FW");
      if (quality_ == TIGHTLEPVETO) push_back("MUF");
    }
    if(version_ == WINTER17PUPPI ){
      push_back("NHF_EC");
      push_back("NEF_FW");
      push_back("NHF_FW");
      push_back("nNeutrals_FW_L");
      push_back("nNeutrals_FW_U");
      if (quality_ == TIGHTLEPVETO) push_back("MUF");
    }
 

    if( (version_ == WINTER17 || version_ == WINTER17PUPPI) && quality_ == LOOSE ){
      edm::LogWarning("BadJetIDVersion") << "Winter17 JetID version does not support the LOOSE operating point -- defaulting to TIGHT";
      quality_ = TIGHT;
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
      else if(version_ == WINTER16){
	set("NHF_EC",0.98);
	set("NEF_EC",0.01);
	set("nNeutrals_EC",2);
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }


    } else if ( quality_ == TIGHT ) {
      set("CHF", 0.0);
      set("NHF", 0.9);
      if(version_ != WINTER17 && version_ != WINTER17PUPPI ) set("CEF", 0.99);
      set("NEF", 0.9);
      set("NCH", 0);
      set("nConstituents", 1);
      if(version_ == RUNIISTARTUP){
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }
      else if(version_ == WINTER16){
	set("NHF_EC",0.98);
	set("NEF_EC",0.01);
	set("nNeutrals_EC",2);
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }
      else if(version_ == WINTER17){
	set("NEF_EC_L",0.02);
	set("NEF_EC_U",0.99);
	set("nNeutrals_EC",2);
	set("NHF_FW",0.02);
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
      }
      else if(version_ == WINTER17PUPPI){
	set("NHF_EC",0.99);
	set("NHF_FW",0.02);
	set("NEF_FW",0.90);
	set("nNeutrals_FW_L",2);
	set("nNeutrals_FW_U",15);
      }

    }else if ( quality_ == TIGHTLEPVETO ) {
      set("CHF", 0.0);
      set("NHF", 0.9);
      set("NEF", 0.9);
      set("NCH", 0);
      set("nConstituents", 1);
      if(version_ == WINTER17){
	set("CEF", 0.8);
	set("NEF_EC_L",0.02);
	set("NEF_EC_U",0.99);
	set("nNeutrals_EC",2);
	set("NHF_FW",0.02);
	set("NEF_FW",0.90);
	set("nNeutrals_FW",10);
	set("MUF", 0.8);
      }
      else if(version_ == WINTER17PUPPI){
	set("CEF", 0.8);
	set("NHF_EC",0.99);
	set("NHF_FW",0.02);
	set("NEF_FW",0.90);
	set("nNeutrals_FW_L",2);
	set("nNeutrals_FW_U",15);
	set("MUF", 0.8);
      }
      else if(version_ == WINTER16){
	set("CEF", 0.9);
	set("NEF_EC",0.01);
	set("NHF_EC",0.98);
	set("nNeutrals_EC",2);
	set("nNeutrals_FW",10);
	set("NEF_FW",0.90);
	set("MUF", 0.8);
      }

    }
  }

  void initIndex()
  {
    indexNConstituents_ = index_type (&bits_, "nConstituents");
    indexNEF_ = index_type (&bits_, "NEF");
    indexNHF_ = index_type (&bits_, "NHF");
    if((version_ != WINTER17 && version_ != WINTER17PUPPI) || quality_ != TIGHT ) indexCEF_ = index_type (&bits_, "CEF");

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
      if ( quality_ == TIGHTLEPVETO ) {indexMUF_ = index_type (&bits_, "MUF");}
    }
    if(version_ == WINTER17){
      indexNEF_EC_L_ = index_type (&bits_, "NEF_EC_L");
      indexNEF_EC_U_ = index_type (&bits_, "NEF_EC_U");
      indexNNeutrals_EC_ = index_type (&bits_, "nNeutrals_EC");
      indexNHF_FW_ = index_type (&bits_, "NHF_FW");
      indexNEF_FW_ = index_type (&bits_, "NEF_FW");
      indexNNeutrals_FW_ = index_type (&bits_, "nNeutrals_FW");
      if ( quality_ == TIGHTLEPVETO ) {indexMUF_ = index_type (&bits_, "MUF");}
    }
    if(version_ == WINTER17PUPPI){
      indexNHF_EC_ = index_type (&bits_, "NHF_EC");
      indexNHF_FW_ = index_type (&bits_, "NHF_FW");
      indexNEF_FW_ = index_type (&bits_, "NEF_FW");
      indexNNeutrals_FW_L_ = index_type (&bits_, "nNeutrals_FW_L");
      indexNNeutrals_FW_U_ = index_type (&bits_, "nNeutrals_FW_U");
      if ( quality_ == TIGHTLEPVETO ) {indexMUF_ = index_type (&bits_, "MUF");}
    }

    retInternal_ = getBitTemplate();
  }

  Version_t version_;
  Quality_t quality_;

  index_type indexNConstituents_;
  index_type indexNEF_;
  index_type indexMUF_;
  index_type indexNHF_;
  index_type indexCEF_;
  index_type indexCHF_;
  index_type indexNCH_;

  index_type indexNHF_FW_;
  index_type indexNEF_FW_;
  index_type indexNNeutrals_FW_;
  index_type indexNNeutrals_FW_L_;
  index_type indexNNeutrals_FW_U_;

  index_type indexNHF_EC_;
  index_type indexNEF_EC_;
  index_type indexNEF_EC_L_;
  index_type indexNEF_EC_U_;
  index_type indexNNeutrals_EC_;


};

#endif
