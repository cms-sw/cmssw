#include "EgammaAnalysis/ElectronTools/interface/VersionedGsfElectronSelector.h"

VersionedGsfElectronSelector::
VersionedGsfElectronSelector( edm::ParameterSet const & parameters ):
  VersionedSelector<reco::GsfElectron>(parameters) {
  verbose_ = false;
  
  Version_t version = N_VERSIONS;
  std::string versionStr = parameters.getParameter<std::string>("version");

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
  
void VersionedGsfElectronSelector::
initialize( Version_t version,
	    double mva,
	    double d0,
	    int nMissingHits,
	    std::string eidUsed,
	    bool convRej,
	    double pfiso )
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

// cuts based on top group L+J synchronization exercise
bool VersionedGsfElectronSelector::
spring11Cuts( const reco::GsfElectron & electron, pat::strbitset & ret)
{  
  return true;
}  
