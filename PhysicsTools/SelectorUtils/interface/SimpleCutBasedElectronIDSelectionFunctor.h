#ifndef PhysicsTools_PatUtils_interface_SimpleCutBasedElectronIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_SimpleCutBasedElectronIDSelectionFunctor_h

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

/*
___________________________________________________________________________________

Description:
^^^^^^^^^^^^
    This is a class that implements the Simple Cut Based Electron 
    Identification cuts.  A more detailed description of the cuts
    and the tuning method can be found on this twiki:
    
    https://twiki.cern.ch/twiki/bin/view/CMS/SimpleCutBasedEleID

    For more information on how to calculate the magnetic field
    look here:

    https://twiki.cern.ch/twiki/bin/viewauth/CMS/ConversionBackgroundRejection
___________________________________________________________________________________

How to use:
^^^^^^^^^^^
    From CMSSW39 onwards you can simply define an instance of this class:

      SimpleCutBasedElectronIDSelectionFunctor patSele95
      (SimpleCutBasedElectronIDSelectionFunctor::relIso95);

    and get the decision with the following method:
      pat::Electron *myElec = .....;
      bool pass = patSele90(*myElec);

    The various options are listed in the enumeration Version_t. There
    is also the option to enter as a constructor argument a PSet
    with your favorite cuts.
___________________________________________________________________________________

    Contacts: Nikolaos Rompotis and Chris Seez
    Nikolaos dot Rompotis at Cern dot ch
    Chris    dot Seez     at Cern dot ch

    Author:    Nikolaos Rompotis
               many thanks to Sal Rappoccio
    Imperial College London
    7 June 2010, first commit for CMSSW_3_6_1_patchX
    11July 2010, implementing the ICHEP Egamma recommendation for 
                 removing the Delta Eta cut in the endcaps
    30Sept 2010, simplification of conversion rejection in CMSSW39X
___________________________________________________________________________________

*/


class SimpleCutBasedElectronIDSelectionFunctor : public Selector<pat::Electron>  {

 public: // interface  
  
  enum Version_t { relIso95=0, cIso95,  relIso90, cIso90, relIso85, cIso85, 
		   relIso80, cIso80,  relIso70, cIso70, relIso60, cIso60, NONE };
  
  SimpleCutBasedElectronIDSelectionFunctor() {}
  
  // initialize it by inserting directly the cut values in a parameter set
  SimpleCutBasedElectronIDSelectionFunctor(edm::ParameterSet const & parameters)
    {
      // get the cuts from the PS
      initialize( parameters.getParameter<Double_t>("trackIso_EB"), 
		  parameters.getParameter<Double_t>("ecalIso_EB"), 
		  parameters.getParameter<Double_t>("hcalIso_EB"), 
		  parameters.getParameter<Double_t>("sihih_EB"), 
		  parameters.getParameter<Double_t>("dphi_EB"), 
		  parameters.getParameter<Double_t>("deta_EB"), 
		  parameters.getParameter<Double_t>("hoe_EB"), 
		  parameters.getParameter<Double_t>("cIso_EB"), 
		  parameters.getParameter<Double_t>("trackIso_EE"), 
		  parameters.getParameter<Double_t>("ecalIso_EE"), 
		  parameters.getParameter<Double_t>("hcalIso_EE"), 
		  parameters.getParameter<Double_t>("sihih_EE"), 
		  parameters.getParameter<Double_t>("dphi_EE"), 
		  parameters.getParameter<Double_t>("deta_EE"), 
		  parameters.getParameter<Double_t>("hoe_EE"), 
		  parameters.getParameter<Double_t>("cIso_EE"), 
		  parameters.getParameter<Int_t>("conversionRejection"), 
		  parameters.getParameter<Int_t>("maxNumberOfExpectedMissingHits"));
      retInternal_ = getBitTemplate();
    }
  // initialize it by using only the version name
  SimpleCutBasedElectronIDSelectionFunctor(Version_t  version)
    {
      if (version == NONE) {
	std::cout << "SimpleCutBasedElectronIDSelectionFunctor: If you want to use version NONE "
		  << "then you have also to provide the selection cuts by yourself " << std::endl;
	std::cout << "SimpleCutBasedElectronIDSelectionFunctor: ID Version is changed to 80cIso "
		  << std::endl;
	version = cIso80;
      }
      initialize(version);
      retInternal_ = getBitTemplate();
    }

  void initialize( Version_t version ) 
  {
    version_ = version;
    // push back the variables
    push_back("trackIso_EB");
    push_back("ecalIso_EB" );
    push_back("hcalIso_EB" );
    push_back("sihih_EB"   );
    push_back("dphi_EB"    );
    push_back("deta_EB"    );
    push_back("hoe_EB"     );
    push_back("cIso_EB"    );
    
    push_back("trackIso_EE");
    push_back("ecalIso_EE" );
    push_back("hcalIso_EE" );
    push_back("sihih_EE"   );
    push_back("dphi_EE"    );
    push_back("deta_EE"    );
    push_back("hoe_EE"     );
    push_back("cIso_EE"    );
    
    push_back("conversionRejection"            );
    push_back("maxNumberOfExpectedMissingHits" );
    
    
    
    
    if (version_ == relIso95) {
      set("trackIso_EB", 1.5e-01);
      set("ecalIso_EB",  2.0e+00);
      set("hcalIso_EB",  1.2e-01);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     8.0e-01);
      set("deta_EB",     7.0e-03);
      set("hoe_EB",      1.5e-01);
      set("cIso_EB",     10000. );
      
      set("trackIso_EE", 8.0e-02);
      set("ecalIso_EE",  6.0e-02);
      set("hcalIso_EE",  5.0e-02);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     7.0e-01);
      set("deta_EE",     1.0e-02);
      set("hoe_EE",      7.0e-02);
      set("cIso_EE",     10000. );
      
      set("conversionRejection",            0);
      set("maxNumberOfExpectedMissingHits", 1);

    }
    else if (version_ == cIso95) {
      set("trackIso_EB", 100000.);
      set("ecalIso_EB",  100000.);
      set("hcalIso_EB",  100000.);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     8.0e-01);
      set("deta_EB",     7.0e-03);
      set("hoe_EB",      1.5e-01);
      set("cIso_EB",     1.5e-01);
      			       					      
      set("trackIso_EE", 100000.);
      set("ecalIso_EE",  100000.);
      set("hcalIso_EE",  100000.);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     7.0e-01);
      set("deta_EE",     1.0e-02);
      set("hoe_EE",      7.0e-02);
      set("cIso_EE",     1.0e-01);
      
      set("conversionRejection",            0);
      set("maxNumberOfExpectedMissingHits", 1);
      
    }
    else if (version_ == relIso90) {
      set("trackIso_EB", 1.2e-01);
      set("ecalIso_EB",  9.0e-02);
      set("hcalIso_EB",  1.0e-01);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     8.0e-01);
      set("deta_EB",     7.0e-03);
      set("hoe_EB",      1.2e-01);
      set("cIso_EB",     10000. );

      set("trackIso_EE", 5.0e-02);
      set("ecalIso_EE",  6.0e-02);
      set("hcalIso_EE",  3.0e-02);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     7.0e-01);
      set("deta_EE",     9.0e-03);
      set("hoe_EE",      5.0e-02);
      set("cIso_EE",     10000. );
      
      set("conversionRejection",            1);
      set("maxNumberOfExpectedMissingHits", 1);
    }
    else if (version_ == cIso90) {
      set("trackIso_EB", 100000.);
      set("ecalIso_EB",  100000.);
      set("hcalIso_EB",  100000.);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     8.0e-01);
      set("deta_EB",     7.0e-03);
      set("hoe_EB",      1.2e-01);
      set("cIso_EB",     1.0e-01);

      set("trackIso_EE", 100000.);
      set("ecalIso_EE",  100000.);
      set("hcalIso_EE",  100000.);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     7.0e-01);
      set("deta_EE",     9.0e-03);
      set("hoe_EE",      5.0e-02);
      set("cIso_EE",     7.0e-02);
      
      set("conversionRejection",            1);
      set("maxNumberOfExpectedMissingHits", 1);
    }
    else if (version_ == relIso85) {
      set("trackIso_EB", 9.0e-02);
      set("ecalIso_EB",  8.0e-02);
      set("hcalIso_EB",  1.0e-01);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     6.0e-02);
      set("deta_EB",     6.0e-03);
      set("hoe_EB",      4.0e-02);
      set("cIso_EB",     10000. );

      set("trackIso_EE", 5.0e-02);
      set("ecalIso_EE",  5.0e-02);
      set("hcalIso_EE",  2.5e-02);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     4.0e-02);
      set("deta_EE",     7.0e-03);
      set("hoe_EE",      2.5e-02);
      set("cIso_EE",     10000. );
      
      set("conversionRejection",            1);
      set("maxNumberOfExpectedMissingHits", 1);
    }
    else if (version_ == cIso85) {
      set("trackIso_EB", 100000.);
      set("ecalIso_EB",  100000.);
      set("hcalIso_EB",  100000.);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     6.0e-02);
      set("deta_EB",     6.0e-03);
      set("hoe_EB",      4.0e-02);
      set("cIso_EB",     9.0e-02);

      set("trackIso_EE", 100000.);
      set("ecalIso_EE",  100000.);
      set("hcalIso_EE",  100000.);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     4.0e-02);
      set("deta_EE",     7.0e-03);
      set("hoe_EE",      2.5e-02);
      set("cIso_EE",     6.0e-02);
      
      set("conversionRejection",            1);
      set("maxNumberOfExpectedMissingHits", 1);
    }
    else if (version_ == relIso80) {
      set("trackIso_EB", 9.0e-02);
      set("ecalIso_EB",  7.0e-02);
      set("hcalIso_EB",  1.0e-01);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     6.0e-02);
      set("deta_EB",     4.0e-03);
      set("hoe_EB",      4.0e-02);
      set("cIso_EB",     100000.);

      set("trackIso_EE", 4.0e-02);
      set("ecalIso_EE",  5.0e-02);
      set("hcalIso_EE",  2.5e-02);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     3.0e-02);
      set("deta_EE",     7.0e-03);
      set("hoe_EE",      2.5e-02);
      set("cIso_EE",     100000.);
      
      set("conversionRejection",            1);
      set("maxNumberOfExpectedMissingHits", 0);
    }
    else if (version_ == cIso80) {
      set("trackIso_EB", 100000.);
      set("ecalIso_EB",  100000.);
      set("hcalIso_EB",  100000.);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     6.0e-02);
      set("deta_EB",     4.0e-03);
      set("hoe_EB",      4.0e-02);
      set("cIso_EB",     7.0e-02);

      set("trackIso_EE", 100000.);
      set("ecalIso_EE",  100000.);
      set("hcalIso_EE",  100000.);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     3.0e-02);
      set("deta_EE",     7.0e-03);
      set("hoe_EE",      2.5e-02);
      set("cIso_EE",     6.0e-02);
      
      set("conversionRejection",            1);
      set("maxNumberOfExpectedMissingHits", 0);
    }
    else if (version_ == relIso70) {
      set("trackIso_EB", 5.0e-02);
      set("ecalIso_EB",  6.0e-02);
      set("hcalIso_EB",  3.0e-02);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     3.0e-02);
      set("deta_EB",     4.0e-03);
      set("hoe_EB",      2.5e-02);
      set("cIso_EB",     100000.);

      set("trackIso_EE", 2.5e-02);
      set("ecalIso_EE",  2.5e-02);
      set("hcalIso_EE",  2.0e-02);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     2.0e-02);
      set("deta_EE",     5.0e-03);
      set("hoe_EE",      2.5e-02);
      set("cIso_EE",     100000.);
      
      set("conversionRejection",            1);
      set("maxNumberOfExpectedMissingHits", 0);
    }
    else if (version_ == cIso70) {
      set("trackIso_EB", 100000.);
      set("ecalIso_EB",  100000.);
      set("hcalIso_EB",  100000.);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     3.0e-02);
      set("deta_EB",     4.0e-03);
      set("hoe_EB",      2.5e-02);
      set("cIso_EB",     4.0e-02);

      set("trackIso_EE", 100000.);
      set("ecalIso_EE",  100000.);
      set("hcalIso_EE",  100000.);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     2.0e-02);
      set("deta_EE",     5.0e-03);
      set("hoe_EE",      2.5e-02);
      set("cIso_EE",     3.0e-02);
      
      set("conversionRejection",            1);
      set("maxNumberOfExpectedMissingHits", 0);
    }
    else if (version_ == relIso60) {
      set("trackIso_EB", 4.0e-02);
      set("ecalIso_EB",  4.0e-02);
      set("hcalIso_EB",  3.0e-02);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     2.5e-02);
      set("deta_EB",     4.0e-03);
      set("hoe_EB",      2.5e-02);
      set("cIso_EB",     100000.);

      set("trackIso_EE", 2.5e-02);
      set("ecalIso_EE",  2.0e-02);
      set("hcalIso_EE",  2.0e-02);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     2.0e-02);
      set("deta_EE",     5.0e-03);
      set("hoe_EE",      2.5e-02);
      set("cIso_EE",     100000.);
      
      set("conversionRejection",            1);
      set("maxNumberOfExpectedMissingHits", 0);
    }
    else if (version_ == cIso60) {
      set("trackIso_EB", 100000.);
      set("ecalIso_EB",  100000.);
      set("hcalIso_EB",  100000.);
      set("sihih_EB",    1.0e-02);
      set("dphi_EB",     2.5e-02);
      set("deta_EB",     4.0e-03);
      set("hoe_EB",      2.5e-02);
      set("cIso_EB",     3.0e-02);

      set("trackIso_EE", 100000.);
      set("ecalIso_EE",  100000.);
      set("hcalIso_EE",  100000.);
      set("sihih_EE",    3.0e-02);
      set("dphi_EE",     2.0e-02);
      set("deta_EE",     5.0e-03);
      set("hoe_EE",      2.5e-02);
      set("cIso_EE",     2.0e-02);
      
      set("conversionRejection",            1);
      set("maxNumberOfExpectedMissingHits", 0);
    }
  }

  void initialize(Double_t trackIso_EB, Double_t ecalIso_EB, Double_t hcalIso_EB,
		  Double_t sihih_EB, Double_t  dphi_EB, Double_t deta_EB, Double_t hoe_EB,
		  Double_t cIso_EB,
		  Double_t trackIso_EE, Double_t ecalIso_EE, Double_t hcalIso_EE,
		  Double_t sihih_EE, Double_t  dphi_EE, Double_t deta_EE, Double_t hoe_EE,
		  Double_t cIso_EE, Int_t conversionRejection, 
		  Int_t maxNumberOfExpectedMissingHits)
  {
    version_ = NONE;
    push_back("trackIso_EB");
    push_back("ecalIso_EB" );
    push_back("hcalIso_EB" );
    push_back("sihih_EB"   );
    push_back("dphi_EB"    );
    push_back("deta_EB"    );
    push_back("hoe_EB"     );
    push_back("cIso_EB"    );
    
    push_back("trackIso_EE");
    push_back("ecalIso_EE" );
    push_back("hcalIso_EE" );
    push_back("sihih_EE"   );
    push_back("dphi_EE"    );
    push_back("deta_EE"    );
    push_back("hoe_EE"     );
    push_back("cIso_EE"    );
    
    push_back("conversionRejection"            );
    push_back("maxNumberOfExpectedMissingHits" );
    
   
    set("trackIso_EB", trackIso_EB);
    set("ecalIso_EB",  ecalIso_EB);
    set("hcalIso_EB",  hcalIso_EB);
    set("sihih_EB",    sihih_EB);
    set("dphi_EB",     dphi_EB);
    set("deta_EB",     deta_EB);
    set("hoe_EB",      hoe_EB);
    set("cIso_EB",     cIso_EB);
    
    set("trackIso_EE", trackIso_EE);
    set("ecalIso_EE",  ecalIso_EE);
    set("hcalIso_EE",  hcalIso_EE);
    set("sihih_EE",    sihih_EE);
    set("dphi_EE",     dphi_EE);
    set("deta_EE",     deta_EE);
    set("hoe_EE",      hoe_EE);
    set("cIso_EE",     cIso_EE);
    
    set("conversionRejection",            conversionRejection);
    set("maxNumberOfExpectedMissingHits", maxNumberOfExpectedMissingHits);
    
  }

  bool operator()( const pat::Electron & electron, pat::strbitset & ret ) 
  {
    // for the time being only Spring10 variable definition
    return spring10Variables(electron, ret);
  }
  using Selector<pat::Electron>::operator();
  // function with the Spring10 variable definitions
  bool spring10Variables( const pat::Electron & electron, pat::strbitset & ret) 
  {
    ret.set(false);
    //
    Double_t eleET = electron.p4().Pt();
    Double_t trackIso = electron.dr03TkSumPt()/eleET;
    Double_t ecalIso = electron.dr03EcalRecHitSumEt()/eleET;
    Double_t hcalIso = electron.dr03HcalTowerSumEt()/eleET;
    Double_t sihih   = electron.sigmaIetaIeta();
    Double_t Dphi    = electron.deltaPhiSuperClusterTrackAtVtx();
    Double_t Deta    = electron.deltaEtaSuperClusterTrackAtVtx();
    Double_t HoE     = electron.hadronicOverEm();
    Double_t cIso    = 0;
    if (electron.isEB()) { cIso = 
	( electron.dr03TkSumPt() + std::max(0.,electron.dr03EcalRecHitSumEt() -1.) 
	  + electron.dr03HcalTowerSumEt() ) / eleET;
    }
    else {
      cIso = ( electron.dr03TkSumPt()+electron.dr03EcalRecHitSumEt()+
	       electron.dr03HcalTowerSumEt()  ) / eleET;
    }
    Int_t innerHits = electron.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    // in 39 conversion rejection variables are accessible from Gsf electron
    Double_t dist = electron.convDist(); // default value is -9999 if conversion partner not found
    Double_t dcot = electron.convDcot(); // default value is -9999 if conversion partner not found
    Bool_t isConv = fabs(dist) < 0.02 && fabs(dcot) < 0.02;
    // now apply the cuts
    if (electron.isEB()) { // BARREL case
      // check the EB cuts
      if ( trackIso   <  cut("trackIso_EB", double()) || ignoreCut("trackIso_EB")) passCut(ret, "trackIso_EB");
      if ( ecalIso    <  cut("ecalIso_EB",  double()) || ignoreCut("ecalIso_EB") ) passCut(ret, "ecalIso_EB");
      if ( hcalIso    <  cut("hcalIso_EB",  double()) || ignoreCut("hcalIso_EB") ) passCut(ret, "hcalIso_EB");
      if ( sihih      <  cut("sihih_EB",    double()) || ignoreCut("sihih_EB")   ) passCut(ret, "sihih_EB");
      if ( fabs(Dphi) <  cut("dphi_EB",     double()) || ignoreCut("dphi_EB")    ) passCut(ret, "dphi_EB");
      if ( fabs(Deta) <  cut("deta_EB",     double()) || ignoreCut("deta_EB")    ) passCut(ret, "deta_EB");
      if ( HoE        <  cut("hoe_EB",      double()) || ignoreCut("hoe_EB")     ) passCut(ret, "hoe_EB");
      if ( cIso       <  cut("cIso_EB",     double()) || ignoreCut("cIso_EB")    ) passCut(ret, "cIso_EB");
      // pass all the EE cuts
      passCut(ret, "trackIso_EE");	
      passCut(ret, "ecalIso_EE");	
      passCut(ret, "hcalIso_EE");	
      passCut(ret, "sihih_EE");	
      passCut(ret, "dphi_EE");	
      passCut(ret, "deta_EE");	
      passCut(ret, "hoe_EE");	
      passCut(ret, "cIso_EE");     
    } else {  // ENDCAPS case
      // check the EE cuts
      if ( trackIso   <  cut("trackIso_EE", double()) || ignoreCut("trackIso_EE")) passCut(ret, "trackIso_EE");
      if ( ecalIso    <  cut("ecalIso_EE",  double()) || ignoreCut("ecalIso_EE") ) passCut(ret, "ecalIso_EE");
      if ( hcalIso    <  cut("hcalIso_EE",  double()) || ignoreCut("hcalIso_EE") ) passCut(ret, "hcalIso_EE");
      if ( sihih      <  cut("sihih_EE",    double()) || ignoreCut("sihih_EE")   ) passCut(ret, "sihih_EE");
      if ( fabs(Dphi) <  cut("dphi_EE",     double()) || ignoreCut("dphi_EE")    ) passCut(ret, "dphi_EE");
      if ( fabs(Deta) <  cut("deta_EE",     double()) || ignoreCut("deta_EE")    ) passCut(ret, "deta_EE");
      if ( HoE        <  cut("hoe_EE",      double()) || ignoreCut("hoe_EE")     ) passCut(ret, "hoe_EE");
      if ( cIso       <  cut("cIso_EE",     double()) || ignoreCut("cIso_EE")    ) passCut(ret, "cIso_EE");     
      // pass all the EB cuts
      passCut(ret, "trackIso_EB");	
      passCut(ret, "ecalIso_EB");	
      passCut(ret, "hcalIso_EB");	
      passCut(ret, "sihih_EB");	
      passCut(ret, "dphi_EB");	
      passCut(ret, "deta_EB");	
      passCut(ret, "hoe_EB");	
      passCut(ret, "cIso_EB");     
    }

    // conversion rejection common for EB and EE
    if ( innerHits  <=  cut("maxNumberOfExpectedMissingHits", int())) 
      passCut(ret, "maxNumberOfExpectedMissingHits");    
    if ( 0==cut("conversionRejection", int()) || isConv==false)
      passCut(ret, "conversionRejection");
    setIgnored(ret);   
    return (bool)ret;
  }



 private: // member variables
  // version of the cuts  
  Version_t version_;
};


#endif
