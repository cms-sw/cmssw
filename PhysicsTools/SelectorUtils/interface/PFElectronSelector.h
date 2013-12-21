#ifndef PhysicsTools_PatUtils_interface_PFElectronSelector_h
#define PhysicsTools_PatUtils_interface_PFElectronSelector_h

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"
#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

class PFElectronSelector : public Selector<pat::Electron> {
  
 public: // interface
  
  bool verbose_;
  
  enum Version_t { TOPPAG, N_VERSIONS };
  
  PFElectronSelector() {}
  
  PFElectronSelector( edm::ParameterSet const & parameters ) {
    
    verbose_ = false;
    
    std::string versionStr = parameters.getParameter<std::string>("version");
    rhoTag_ = parameters.getParameter< edm::InputTag> ("rhoSrc");
    
    Version_t version = N_VERSIONS;
    
    if ( versionStr == "TOPPAG" ) {
      version = TOPPAG;
    }
    else {
      throw cms::Exception("InvalidInput") << "Expect version to be one of : TOPPAG" << std::endl;
    }


    initialize( version, 
		parameters.getParameter<bool>  ("Fiducial"),
		parameters.getParameter<int>   ("MaxMissingHits"),
		parameters.getParameter<double>("D0"),
		parameters.getParameter<bool>  ("ConversionRejection"),
		parameters.getParameter<double>("PFIso"),
		parameters.getParameter<double>("MVA")
		);
    if ( parameters.exists("cutsToIgnore") )
      setIgnoredCuts( parameters.getParameter<std::vector<std::string> >("cutsToIgnore") );
    
    retInternal_ = getBitTemplate();

  }
  
  void initialize( Version_t version,
		   bool fid=true,
		   int nMissingHits = 0,
		   double d0 = 0.02,
		   bool convRej = true,
		   double pfiso = 0.10,
		   double mva = 0.4
		   )
  {
    version_ = version; 
    
    //    size_t found;
    //    found = eidUsed.find("NONE");
    //  if ( found != string::npos)

    push_back("Fiducial" );
    push_back("MaxMissingHits", nMissingHits  );
    push_back("D0",        d0     );
    push_back("ConversionRejection" );
    push_back("PFIso",    pfiso );
    push_back("MVA",       mva   );

    set("Fiducial", fid);
    set("MaxMissingHits");
    set("D0");
    set("ConversionRejection", convRej);
    set("PFIso");   
    set("MVA");

    indexFid_            = index_type(&bits_, "Fiducial" );
    indexMaxMissingHits_ = index_type(&bits_, "MaxMissingHits" );
    indexD0_             = index_type(&bits_, "D0"           );
    indexConvRej_        = index_type(&bits_, "ConversionRejection" );
    indexPFIso_          = index_type(&bits_, "PFIso"       );
    indexMVA_            = index_type(&bits_, "MVA"         );
  }

  // Allow for multiple definitions of the cuts. 
  bool operator()( const pat::Electron & electron, edm::EventBase const & event, pat::strbitset & ret ) 
  { 
    if (version_ == TOPPAG ) return topPAGRefCuts(electron, event, ret);
    else {
      return false;
    }
  }

  using Selector<pat::Electron>::operator();
  
  // cuts based on top group L+J synchronization exercise
  bool topPAGRefCuts( const pat::Electron & electron, edm::EventBase const & event, pat::strbitset & ret)
  {



    ret.set(false);

    edm::Handle<double> rhoHandle;
    event.getByLabel(rhoTag_, rhoHandle);

    double rhoIso = std::max(*(rhoHandle.product()), 0.0);

    double scEta   = electron.superCluster()->eta();
    double dB      = electron.dB();
    double AEff    = ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaAndNeutralHadronIso03, scEta, ElectronEffectiveArea::kEleEAData2012);
    double chIso = electron.userIsolation(pat::PfChargedHadronIso);
    double nhIso = electron.userIsolation(pat::PfNeutralHadronIso);
    double phIso  = electron.userIsolation(pat::PfGammaIso);
    double pfIso = ( chIso + max(0.0, nhIso + phIso - rhoIso*AEff) )/ electron.ecalDrivenMomentum().pt();
    int mHits  =  electron.gsfTrack()->trackerExpectedHitsInner().numberOfHits();   
    double mva = electron.electronID("mvaTrigV0");
    //Electron Selection for e+jets
    //-----------------------------  
    bool fid = ! (fabs(scEta) > 1.4442 &&  fabs(scEta) < 1.5660 );

		

    if ( fid                                                   || ignoreCut(indexFid_)              ) passCut(ret, indexFid_     );
    if ( mHits         <= cut(indexMaxMissingHits_,  double()) || ignoreCut(indexMaxMissingHits_)   ) passCut(ret, indexMaxMissingHits_  );
    if ( fabs(dB)      <  cut(indexD0_,              double()) || ignoreCut(indexD0_)               ) passCut(ret, indexD0_     );
    if ( electron.passConversionVeto()                         || ignoreCut(indexConvRej_)          ) passCut(ret, indexConvRej_     );
    if ( pfIso         <  cut(indexPFIso_,           double()) || ignoreCut(indexPFIso_)            ) passCut(ret, indexPFIso_ );
    if ( mva           >  cut(indexMVA_,             double()) || ignoreCut(indexMVA_)              ) passCut(ret, indexMVA_ );
    setIgnored(ret);
    return (bool)ret;
  }
  
 private: // member variables
  
  Version_t version_;

  index_type indexFid_;
  index_type indexMaxMissingHits_;
  index_type indexD0_;
  index_type indexConvRej_;
  index_type indexPFIso_;
  index_type indexMVA_;


  std::string electronIDvalue_;
  edm::InputTag rhoTag_;

  bool operator()( const pat::Electron & electron, pat::strbitset & ret ) { return false;}
};

#endif
