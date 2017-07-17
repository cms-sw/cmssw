#ifndef PhysicsTools_PatUtils_interface_ElectronVPlusJetsIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_ElectronVPlusJetsIDSelectionFunctor_h

#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ElectronVPlusJetsIDSelectionFunctor : public Selector<pat::Electron>  {

 public: // interface

  enum Version_t { SUMMER08, FIRSTDATA, N_VERSIONS };

  ElectronVPlusJetsIDSelectionFunctor() {}

#ifndef __GCCXML__
  ElectronVPlusJetsIDSelectionFunctor( edm::ParameterSet const & parameters, edm::ConsumesCollector& iC ) :
    ElectronVPlusJetsIDSelectionFunctor(parameters)
  {}
#endif

  ElectronVPlusJetsIDSelectionFunctor( edm::ParameterSet const & parameters ){

    std::string versionStr = parameters.getParameter<std::string>("version");
    if ( versionStr != "FIRSTDATA") {
      std::cout << "The version " << versionStr << " is deprecated. Setting to FIRSTDATA" << std::endl;
    }

    if (versionStr == "FIRSTDATA") {
      initialize( FIRSTDATA,
		  parameters.getParameter<double>("D0"),
		  parameters.getParameter<double>("ED0"),
		  parameters.getParameter<double>("SD0"),
		  parameters.getParameter<double>("RelIso") );
      if ( parameters.exists("cutsToIgnore") )
	setIgnoredCuts( parameters.getParameter<std::vector<std::string> >("cutsToIgnore") );
    } else {
      throw cms::Exception("InvalidInput") << "Expect version to be one of SUMMER08, FIRSTDATA," << std::endl;
    }

    retInternal_ = getBitTemplate();

  }


  ElectronVPlusJetsIDSelectionFunctor( Version_t version,
				       double d0 = 0.2,
				       double ed0 = 999.0,
				       double sd0 = 999.0,
				       double reliso = 0.1) {
    initialize( version, d0, ed0, sd0, reliso );
  }


  void initialize( Version_t version,
		   double d0,
		   double ed0,
		   double sd0,
		   double reliso)
  {
    version_ = version;

    push_back("D0",        d0);
    push_back("ED0",       ed0);
    push_back("SD0",       sd0);
    push_back("RelIso",    reliso);

    // all on by default
    set("D0");
    set("ED0");
    set("SD0");
    set("RelIso");


    indexD0_            = index_type(&bits_, "D0"           );
    indexED0_           = index_type(&bits_, "ED0"          );
    indexSD0_           = index_type(&bits_, "SD0"          );
    indexRelIso_        = index_type(&bits_, "RelIso"       );

  }

  // Allow for multiple definitions of the cuts.
  bool operator()( const pat::Electron & electron, pat::strbitset & ret )
  {
    if ( version_ == FIRSTDATA ) return firstDataCuts( electron, ret );
    else {
      return false;
    }
  }

  using Selector<pat::Electron>::operator();

  // cuts based on craft 08 analysis.
  bool firstDataCuts( const pat::Electron & electron, pat::strbitset & ret)
  {

    ret.set(false);
    double corr_d0 = electron.dB();
    double corr_ed0 = electron.edB();
    double corr_sd0 = ( corr_ed0 > 0.000000001 ) ? corr_d0 / corr_ed0 : 999.0;

    double hcalIso = electron.dr03HcalTowerSumEt();
    double ecalIso = electron.dr03EcalRecHitSumEt();
    double trkIso  = electron.dr03TkSumPt();
    double et      = electron.et() ;

    double relIso = (ecalIso + hcalIso + trkIso) / et;

    if ( fabs(corr_d0) <  cut(indexD0_,     double()) || ignoreCut(indexD0_)      ) passCut(ret, indexD0_     );
    if ( fabs(corr_ed0)<  cut(indexED0_,    double()) || ignoreCut(indexED0_)     ) passCut(ret, indexED0_    );
    if ( fabs(corr_sd0)<  cut(indexSD0_,    double()) || ignoreCut(indexSD0_)     ) passCut(ret, indexSD0_    );
    if ( relIso        <  cut(indexRelIso_, double()) || ignoreCut(indexRelIso_)  ) passCut(ret, indexRelIso_ );

    setIgnored(ret);
    return (bool)ret;
  }


 private: // member variables

  Version_t version_;

  index_type indexD0_;
  index_type indexED0_;
  index_type indexSD0_;
  index_type indexRelIso_;

};

#endif
