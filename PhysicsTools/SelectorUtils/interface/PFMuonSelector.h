#ifndef PhysicsTools_PatUtils_interface_PFMuonSelector_h
#define PhysicsTools_PatUtils_interface_PFMuonSelector_h

#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

class PFMuonSelector : public Selector<pat::Muon> {

 public: // interface

  bool verbose_;

  enum Version_t { TOPPAG12_LJETS, N_VERSIONS };

  PFMuonSelector() {}

#ifndef __GCCXML__
  PFMuonSelector( edm::ParameterSet const & parameters, edm::ConsumesCollector&& iC ) :
    PFMuonSelector( parameters )
  {}
#endif

  PFMuonSelector( edm::ParameterSet const & parameters ) {

    verbose_ = false;

    std::string versionStr = parameters.getParameter<std::string>("version");

    Version_t version = N_VERSIONS;

    if ( versionStr == "TOPPAG12_LJETS" ) {
      version = TOPPAG12_LJETS;
    }
    else {
      throw cms::Exception("InvalidInput") << "Expect version to be one of SPRING11" << std::endl;
    }

    initialize( version,
                parameters.getParameter<double>("Chi2"),
                parameters.getParameter<int>   ("minTrackerLayers"),
                parameters.getParameter<int>   ("minValidMuHits"),
                parameters.getParameter<double>("maxIp"),
                parameters.getParameter<int>   ("minPixelHits"),
                parameters.getParameter<int>   ("minMatchedStations"),
                parameters.getParameter<double>("maxPfRelIso")
		);
    if ( parameters.exists("cutsToIgnore") )
      setIgnoredCuts( parameters.getParameter<std::vector<std::string> >("cutsToIgnore") );

    retInternal_ = getBitTemplate();

  }

  void initialize( Version_t version,
                   double    chi2             = 10.0,
                   int       minTrackerLayers = 6,
                   int       minValidMuonHits = 1,
                   double    maxIp            = 0.2,
                   int       minPixelHits     = 1,
                   int       minNMatches      = 2,
                   double    pfiso            = 0.12
                   )
  {
    version_ = version;

    push_back("GlobalMuon",         true);
    push_back("TrackerMuon",        true);
    push_back("Chi2",               chi2   );
    push_back("minTrackerLayers",   minTrackerLayers);
    push_back("minValidMuHits",     minValidMuonHits  );
    push_back("maxIp",              maxIp );
    push_back("minPixelHits",       minPixelHits);
    push_back("minMatchedStations", minNMatches);
    push_back("maxPfRelIso",        pfiso );

    set("GlobalMuon");
    set("TrackerMuon");
    set("Chi2");
    set("minTrackerLayers");
    set("minValidMuHits");
    set("maxIp");
    set("minPixelHits");
    set("minMatchedStations");
    set("maxPfRelIso");

    indexChi2_             = index_type(&bits_, "Chi2"            );
    indexMinTrackerLayers_ = index_type(&bits_, "minTrackerLayers" );
    indexminValidMuHits_   = index_type(&bits_, "minValidMuHits"      );
    indexMaxIp_            = index_type(&bits_, "maxIp"      );
    indexPixHits_          = index_type(&bits_, "minPixelHits"      );
    indexStations_         = index_type(&bits_, "minMatchedStations");
    indexmaxPfRelIso_      = index_type(&bits_, "maxPfRelIso"           );


    if (version_ == TOPPAG12_LJETS ){
      set("TrackerMuon", false);
    }

  }

  // Allow for multiple definitions of the cuts.
  bool operator()( const pat::Muon & muon, pat::strbitset & ret )
  {
    if (version_ == TOPPAG12_LJETS ) return TopPag12LjetsCuts(muon, ret);
    else {
      return false;
    }
  }

  using Selector<pat::Muon>::operator();

  bool TopPag12LjetsCuts( const pat::Muon & muon, pat::strbitset & ret){

    ret.set(false);

    bool isGlobal  = muon.isGlobalMuon();
    bool isTracker = muon.isTrackerMuon();

    double norm_chi2     = 9999999.0;
    int minTrackerLayers = 0;
    int minValidMuonHits = 0;
    int _ip = 0.0;
    int minPixelHits = 0;
    if ( muon.globalTrack().isNonnull() && muon.globalTrack().isAvailable() ){
      norm_chi2        = muon.normChi2();
      minTrackerLayers = static_cast<int> (muon.track()->hitPattern().trackerLayersWithMeasurement());
      minValidMuonHits = static_cast<int> (muon.globalTrack()->hitPattern().numberOfValidMuonHits());
      _ip = muon.dB();
      minPixelHits = muon.innerTrack()->hitPattern().numberOfValidPixelHits();
    }

    int minMatchedStations = muon.numberOfMatches();

    double chIso = muon.userIsolation(pat::PfChargedHadronIso);
    double nhIso = muon.userIsolation(pat::PfNeutralHadronIso);
    double gIso  = muon.userIsolation(pat::PfGammaIso);
    double pt    = muon.pt() ;

    double pfIso = (chIso + nhIso + gIso) / pt;

    if ( isGlobal  || ignoreCut("GlobalMuon")  )  passCut(ret, "GlobalMuon" );
    if ( isTracker || ignoreCut("TrackerMuon")  )  passCut(ret, "TrackerMuon" );
    if ( norm_chi2          <  cut(indexChi2_,   double()) || ignoreCut(indexChi2_)    ) passCut(ret, indexChi2_   );
    if ( minTrackerLayers   >= cut(indexMinTrackerLayers_,int()) || ignoreCut(indexMinTrackerLayers_)) passCut(ret, indexMinTrackerLayers_  );
    if ( minValidMuonHits   >= cut(indexminValidMuHits_,int()) || ignoreCut(indexminValidMuHits_)) passCut(ret, indexminValidMuHits_  );
    if ( _ip                <  cut(indexMaxIp_,double()) || ignoreCut(indexMaxIp_)) passCut(ret, indexMaxIp_  );
    if ( minPixelHits       >= cut(indexPixHits_,int())    || ignoreCut(indexPixHits_))  passCut(ret, indexPixHits_);
    if ( minMatchedStations >= cut(indexStations_,int())  || ignoreCut(indexStations_))  passCut(ret, indexStations_);
    if ( pfIso              <  cut(indexmaxPfRelIso_, double())  || ignoreCut(indexmaxPfRelIso_)  ) passCut(ret, indexmaxPfRelIso_ );

    setIgnored(ret);

    return (bool)ret;
  }



 private: // member variables

  Version_t version_;

  index_type indexChi2_;
  index_type indexMinTrackerLayers_;
  index_type indexminValidMuHits_;
  index_type indexMaxIp_;
  index_type indexPixHits_;
  index_type indexStations_;
  index_type indexmaxPfRelIso_;


};

#endif
