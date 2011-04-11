#ifndef PhysicsTools_PatUtils_interface_PFElectronSelector_h
#define PhysicsTools_PatUtils_interface_PFElectronSelector_h

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

class PFElectronSelector : public Selector<pat::Electron> {

 public: // interface

  bool verbose_;
  
  enum Version_t { SPRING11, N_VERSIONS };

  PFElectronSelector() {}

  PFElectronSelector( edm::ParameterSet const & parameters ) {

    verbose_ = false;
    
    std::string versionStr = parameters.getParameter<std::string>("version");

    Version_t version = N_VERSIONS;

    if ( versionStr == "SPRING11" ) {
      version = SPRING11;
    }
    else {
      throw cms::Exception("InvalidInput") << "Expect version to be one of SPRING11" << std::endl;
    }

    initialize( version, 
		parameters.getParameter<double>("MVA"),
		parameters.getParameter<double>("D0")  ,
		parameters.getParameter<int>   ("MaxMissingHits")   ,
		parameters.getParameter<double>("PFIso")
		);
    if ( parameters.exists("cutsToIgnore") )
      setIgnoredCuts( parameters.getParameter<std::vector<std::string> >("cutsToIgnore") );
	
    retInternal_ = getBitTemplate();

  }

  void initialize( Version_t version,
		   double mva = 0.4,
		   double d0 = 0.02,
		   int nMissingHits = 1,
		   double pfiso = 0.15 )
  {
    version_ = version; 

    push_back("MVA",       mva   );
    push_back("D0",        d0     );
    push_back("MaxMissingHits", nMissingHits  );
    push_back("PFIso",    pfiso );

    set("MVA");
    set("D0");
    set("MaxMissingHits");
    set("PFIso");   

    indexMVA_            = index_type(&bits_, "MVA"          );
    indexD0_             = index_type(&bits_, "D0"           );
    indexMaxMissingHits_ = index_type(&bits_, "MaxMissingHits" );
    indexPFIso_          = index_type(&bits_, "PFIso"       );
  }

  // Allow for multiple definitions of the cuts. 
  bool operator()( const pat::Electron & electron, pat::strbitset & ret ) 
  { 
    if (version_ == SPRING11 ) return spring11Cuts(electron, ret);
    else {
      return false;
    }
  }

  using Selector<pat::Electron>::operator();

  // cuts based on top group L+J synchronization exercise
  bool spring11Cuts( const pat::Electron & electron, pat::strbitset & ret)
  {

    ret.set(false);

    double mva = electron.mva();
    double missingHits = electron.gsfTrack()->trackerExpectedHitsInner().numberOfHits() ;
    double corr_d0 = electron.dB();

    double chIso = electron.userIsolation(pat::PfChargedHadronIso);
    double nhIso = electron.userIsolation(pat::PfNeutralHadronIso);
    double gIso  = electron.userIsolation(pat::PfGammaIso);
    double et    = electron.et() ;

    double pfIso = (chIso + nhIso + gIso) / et;

    if ( mva           >  cut(indexMVA_,             double()) || ignoreCut(indexMVA_)              ) passCut(ret, indexMVA_ );
    if ( missingHits   <= cut(indexMaxMissingHits_,  double()) || ignoreCut(indexMaxMissingHits_)   ) passCut(ret, indexMaxMissingHits_  );
    if ( fabs(corr_d0) <  cut(indexD0_,              double()) || ignoreCut(indexD0_)               ) passCut(ret, indexD0_     );
    if ( pfIso         <  cut(indexPFIso_,           double()) || ignoreCut(indexPFIso_)            ) passCut(ret, indexPFIso_ );

    setIgnored(ret);
    return (bool)ret;
  }

  
  
 private: // member variables
  
  Version_t version_;

  index_type indexMVA_;
  index_type indexMaxMissingHits_;
  index_type indexD0_;
  index_type indexPFIso_;


};

#endif
