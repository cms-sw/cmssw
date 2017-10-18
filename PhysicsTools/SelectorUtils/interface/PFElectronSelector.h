#ifndef PhysicsTools_PatUtils_interface_PFElectronSelector_h
#define PhysicsTools_PatUtils_interface_PFElectronSelector_h

#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
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

#ifndef __GCCXML__
  PFElectronSelector( edm::ParameterSet const & parameters, edm::ConsumesCollector&& iC ) :
    PFElectronSelector( parameters )
  {}
#endif

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
		parameters.getParameter<int>   ("MaxMissingHits"),
		parameters.getParameter<std::string> ("electronIDused"),
		parameters.getParameter<bool>  ("ConversionRejection"),
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
		   std::string eidUsed = "eidTightMC",
		   bool convRej = true,
		   double pfiso = 0.15 )
  {
    version_ = version;

    //    size_t found;
    //    found = eidUsed.find("NONE");
    //  if ( found != string::npos)
    electronIDvalue_ = eidUsed;

    push_back("D0",        d0     );
    push_back("MaxMissingHits", nMissingHits  );
    push_back("electronID");
    push_back("ConversionRejection" );
    push_back("PFIso",    pfiso );
    push_back("MVA",       mva   );

    set("D0");
    set("MaxMissingHits");
    set("electronID");
    set("ConversionRejection", convRej);
    set("PFIso");
    set("MVA");

    indexD0_             = index_type(&bits_, "D0"           );
    indexMaxMissingHits_ = index_type(&bits_, "MaxMissingHits" );
    indexElectronId_     = index_type(&bits_, "electronID" );
    indexConvRej_        = index_type(&bits_, "ConversionRejection" );
    indexPFIso_          = index_type(&bits_, "PFIso"       );
    indexMVA_            = index_type(&bits_, "MVA"         );
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

    double mva = electron.mva_e_pi();
    double missingHits = electron.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    double corr_d0 = electron.dB();

    // in >= 39x conversion rejection variables are accessible from Gsf electron
    Double_t dist = electron.convDist(); // default value is -9999 if conversion partner not found
    Double_t dcot = electron.convDcot(); // default value is -9999 if conversion partner not found
    bool isNotConv = !(fabs(dist) < 0.02 && fabs(dcot) < 0.02);

    int bitWiseResults =  (int) electron.electronID( electronIDvalue_ );
    bool electronIDboolean = ((bitWiseResults & 1) == 1 );

    double chIso = electron.userIsolation(pat::PfChargedHadronIso);
    double nhIso = electron.userIsolation(pat::PfNeutralHadronIso);
    double gIso  = electron.userIsolation(pat::PfGammaIso);
    double et    = electron.et() ;

    double pfIso = (chIso + nhIso + gIso) / et;

    if ( missingHits   <= cut(indexMaxMissingHits_,  double()) || ignoreCut(indexMaxMissingHits_)   ) passCut(ret, indexMaxMissingHits_  );
    if ( fabs(corr_d0) <  cut(indexD0_,              double()) || ignoreCut(indexD0_)               ) passCut(ret, indexD0_     );
    if ( isNotConv                                             || ignoreCut(indexConvRej_)          ) passCut(ret, indexConvRej_     );
    if ( pfIso         <  cut(indexPFIso_,           double()) || ignoreCut(indexPFIso_)            ) passCut(ret, indexPFIso_ );
    if ( mva           >  cut(indexMVA_,             double()) || ignoreCut(indexMVA_)              ) passCut(ret, indexMVA_ );
    if ( electronIDboolean                                     || ignoreCut(indexElectronId_)       ) passCut(ret, indexElectronId_);
    setIgnored(ret);
    return (bool)ret;
  }

 private: // member variables

  Version_t version_;

  index_type indexID;
  index_type indexMaxMissingHits_;
  index_type indexD0_;
  index_type indexConvRej_;
  index_type indexPFIso_;
  index_type indexMVA_;
  index_type indexElectronId_;

  std::string electronIDvalue_;
};

#endif
