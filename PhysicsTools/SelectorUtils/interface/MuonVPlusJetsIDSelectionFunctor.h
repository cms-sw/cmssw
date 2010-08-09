#ifndef PhysicsTools_PatUtils_interface_MuonVPlusJetsIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_MuonVPlusJetsIDSelectionFunctor_h

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

class MuonVPlusJetsIDSelectionFunctor : public Selector<pat::Muon> {

 public: // interface

  enum Version_t { SUMMER08, FIRSTDATA, SPRING10, N_VERSIONS };

  MuonVPlusJetsIDSelectionFunctor() {}

  MuonVPlusJetsIDSelectionFunctor( edm::ParameterSet const & parameters ) {

    std::string versionStr = parameters.getParameter<std::string>("version");

    Version_t version = N_VERSIONS;

    if ( versionStr == "SUMMER08" ) {
      version = SUMMER08;
    }
    else if ( versionStr == "FIRSTDATA" ) {
      version = FIRSTDATA;
    }
    else if ( versionStr == "SPRING10" ) {
      version = SPRING10;
    }
    else {
      throw cms::Exception("InvalidInput") << "Expect version to be one of SUMMER08, FIRSTDATA, SPRING10," << std::endl;
    }

    initialize( version, 
		parameters.getParameter<double>("Chi2"),
		parameters.getParameter<double>("D0")  ,
		parameters.getParameter<double>("ED0")  ,
		parameters.getParameter<double>("SD0")  ,
		parameters.getParameter<int>   ("NHits")   ,
		parameters.getParameter<int>   ("NValMuHits"),
		parameters.getParameter<double>("ECalVeto")   ,
		parameters.getParameter<double>("HCalVeto")   ,
		parameters.getParameter<double>("RelIso") );
    if ( parameters.exists("cutsToIgnore") )
      setIgnoredCuts( parameters.getParameter<std::vector<std::string> >("cutsToIgnore") );
	
    retInternal_ = getBitTemplate();

    recalcDBFromBSp_ = parameters.getParameter<bool>("RecalcFromBeamSpot");
    beamLineSrc_ = parameters.getParameter<edm::InputTag>("beamLineSrc");    
  }


  MuonVPlusJetsIDSelectionFunctor( Version_t version,
				   double chi2 = 10.0,
				   double d0 = 0.2,
				   double ed0 = 999.0,
				   double sd0 = 999.0,
				   int nhits = 11,
				   int nValidMuonHits = 0,
				   double ecalveto = 4.0,
				   double hcalveto = 6.0,
				   double reliso = 0.05
				   ) : recalcDBFromBSp_(false) {
    initialize( version, chi2, d0, ed0, sd0, nhits, nValidMuonHits, ecalveto, hcalveto, reliso );
  }
  
  


  void initialize( Version_t version,
		   double chi2 = 10.0,
		   double d0 = 999.0,
		   double ed0 = 999.0,
		   double sd0 = 3.0,
		   int nhits = 11,
		   int nValidMuonHits = 0,
		   double ecalveto = 4.0,
		   double hcalveto = 6.0,
		   double reliso = 0.05 )
  {
    version_ = version; 

    push_back("Chi2",      chi2   );
    push_back("D0",        d0     );
    push_back("ED0",       ed0    );
    push_back("SD0",       sd0    );
    push_back("NHits",     nhits  );
    push_back("NValMuHits",     nValidMuonHits  );
    push_back("ECalVeto",  ecalveto);
    push_back("HCalVeto",  hcalveto);
    push_back("RelIso",    reliso );

    set("Chi2");
    set("D0");
    set("ED0");
    set("SD0");
    set("NHits");
    set("NValMuHits");
    set("ECalVeto");
    set("HCalVeto");
    set("RelIso");

    if ( version == SPRING10) {
      set("ED0", false );
      set("SD0", false);
      set("ECalVeto",false);
      set("HCalVeto",false);
    } else if ( version_ == FIRSTDATA ) {
      set("D0", false );
      set("ED0", false );
      set("NValMuHits",false);
    } else if (version == SUMMER08 ) {
      set("SD0", false);
      set("NValMuHits",false);      
    }

  }

  // Allow for multiple definitions of the cuts. 
  bool operator()( const pat::Muon & muon, edm::EventBase const & event, pat::strbitset & ret ) 
  { 

    if (version_ == SPRING10 ) return spring10Cuts(muon, event, ret);
    else if ( version_ == SUMMER08 ) return summer08Cuts( muon, ret );
    else if ( version_ == FIRSTDATA ) return firstDataCuts( muon, ret );
    else {
      return false;
    }
  }

  // For compatibility with older versions.
  bool operator()( const pat::Muon & muon, pat::strbitset & ret ) 
  { 

    if (version_ == SPRING10 ) throw cms::Exception("LogicError") 
      << "MuonVPlusJetsSelectionFunctor SPRING10 version needs the event! Call operator()(muon,event,ret)"
      <<std::endl;

    else if ( version_ == SUMMER08 ) return summer08Cuts( muon, ret );
    else if ( version_ == FIRSTDATA ) return firstDataCuts( muon, ret );
    else {
      return false;
    }
  }


  using Selector<pat::Muon>::operator();

  // cuts based on craft 08 analysis. 
  bool summer08Cuts( const pat::Muon & muon, pat::strbitset & ret)
  {

    ret.set(false);

    double norm_chi2 = muon.normChi2();
    double corr_d0 = muon.dB();
    int nhits = static_cast<int>( muon.numberOfValidHits() );
    
    double ecalVeto = muon.isolationR03().emVetoEt;
    double hcalVeto = muon.isolationR03().hadVetoEt;
	
    double hcalIso = muon.hcalIso();
    double ecalIso = muon.ecalIso();
    double trkIso  = muon.trackIso();
    double pt      = muon.pt() ;

    double relIso = (ecalIso + hcalIso + trkIso) / pt;

    if ( norm_chi2     <  cut("Chi2",   double()) || ignoreCut("Chi2")    ) passCut(ret, "Chi2"   );
    if ( fabs(corr_d0) <  cut("D0",     double()) || ignoreCut("D0")      ) passCut(ret, "D0"     );
    if ( nhits         >= cut("NHits",  int()   ) || ignoreCut("NHits")   ) passCut(ret, "NHits"  );
    if ( hcalVeto      <  cut("HCalVeto",double())|| ignoreCut("HCalVeto")) passCut(ret, "HCalVeto");
    if ( ecalVeto      <  cut("ECalVeto",double())|| ignoreCut("ECalVeto")) passCut(ret, "ECalVeto");
    if ( relIso        <  cut("RelIso", double()) || ignoreCut("RelIso")  ) passCut(ret, "RelIso" );

    setIgnored(ret);

    return (bool)ret;
  }



  // cuts based on craft 08 analysis. 
  bool firstDataCuts( const pat::Muon & muon, pat::strbitset & ret)
  {

    ret.set(false);

    double norm_chi2 = muon.normChi2();
    double corr_d0 = muon.dB();
    double corr_ed0 = muon.edB();
    double corr_sd0 = ( corr_ed0 > 0.000000001 ) ? corr_d0 / corr_ed0 : 999.0;
    int nhits = static_cast<int>( muon.numberOfValidHits() );
    
    double ecalVeto = muon.isolationR03().emVetoEt;
    double hcalVeto = muon.isolationR03().hadVetoEt;
	
    double hcalIso = muon.hcalIso();
    double ecalIso = muon.ecalIso();
    double trkIso  = muon.trackIso();
    double pt      = muon.pt() ;

    double relIso = (ecalIso + hcalIso + trkIso) / pt;

    if ( norm_chi2     <  cut("Chi2",   double()) || ignoreCut("Chi2")    ) passCut(ret, "Chi2"   );
    if ( fabs(corr_d0) <  cut("D0",     double()) || ignoreCut("D0")      ) passCut(ret, "D0"     );
    if ( fabs(corr_ed0)<  cut("ED0",    double()) || ignoreCut("ED0")     ) passCut(ret, "ED0"    );
    if ( fabs(corr_sd0)<  cut("SD0",    double()) || ignoreCut("SD0")     ) passCut(ret, "SD0"    );
    if ( nhits         >= cut("NHits",  int()   ) || ignoreCut("NHits")   ) passCut(ret, "NHits"  );
    if ( hcalVeto      <  cut("HCalVeto",double())|| ignoreCut("HCalVeto")) passCut(ret, "HCalVeto");
    if ( ecalVeto      <  cut("ECalVeto",double())|| ignoreCut("ECalVeto")) passCut(ret, "ECalVeto");
    if ( relIso        <  cut("RelIso", double()) || ignoreCut("RelIso")  ) passCut(ret, "RelIso" );

    setIgnored(ret);
    
    return (bool)ret;
  }

  // cuts based on top group L+J synchronization exercise
  bool spring10Cuts( const pat::Muon & muon, edm::EventBase const & event, pat::strbitset & ret)
  {

    ret.set(false);

    double norm_chi2 = muon.normChi2();
    double corr_d0 = muon.dB();
    double corr_ed0 = muon.edB();
    double corr_sd0 = ( corr_ed0 > 0.000000001 ) ? corr_d0 / corr_ed0 : 999.0;

    //If required, recalculate the impact parameter using the beam spot
    if (recalcDBFromBSp_) {

      //Get the beam spot
      reco::TrackBase::Point beamPoint(0,0,0);
      reco::BeamSpot beamSpot;
      edm::Handle<reco::BeamSpot> beamSpotHandle;
      event.getByLabel(beamLineSrc_, beamSpotHandle);
      
      if( beamSpotHandle.isValid() ){
	beamSpot = *beamSpotHandle;
      } else{
	edm::LogError("DataNotAvailable")
	  << "No beam spot available from EventSetup, not adding high level selection \n";
      }
      beamPoint = reco::TrackBase::Point ( beamSpot.x0(), beamSpot.y0(), beamSpot.z0() );
      
      //Use the beamspot to correct the impact parameter and uncertainty
      reco::TrackRef innerTrack = muon.innerTrack();
      if ( innerTrack.isNonnull() && innerTrack.isAvailable() ) {
	corr_d0 = -1.0 * innerTrack->dxy( beamPoint );
	corr_ed0 = sqrt( innerTrack->d0Error() * innerTrack->d0Error() 
			 + 0.5* beamSpot.BeamWidthX()*beamSpot.BeamWidthX() 
			 + 0.5* beamSpot.BeamWidthY()*beamSpot.BeamWidthY() );
	corr_sd0 = ( corr_ed0 > 0.000000001 ) ? corr_d0 / corr_ed0 : 999.0;

      } else {
	corr_d0 =  999.;
	corr_ed0 = 999.;
      }
    }

    int nhits = static_cast<int>( muon.numberOfValidHits() );
    int nValidMuonHits = static_cast<int> (muon.globalTrack()->hitPattern().numberOfValidMuonHits());
    
    double ecalVeto = muon.isolationR03().emVetoEt;
    double hcalVeto = muon.isolationR03().hadVetoEt;
	
    double hcalIso = muon.hcalIso();
    double ecalIso = muon.ecalIso();
    double trkIso  = muon.trackIso();
    double pt      = muon.pt() ;

    double relIso = (ecalIso + hcalIso + trkIso) / pt;

    if ( norm_chi2     <  cut("Chi2",   double()) || ignoreCut("Chi2")    ) passCut(ret, "Chi2"   );
    if ( fabs(corr_d0) <  cut("D0",     double()) || ignoreCut("D0")      ) passCut(ret, "D0"     );
    if ( fabs(corr_ed0)<  cut("ED0",    double()) || ignoreCut("ED0")     ) passCut(ret, "ED0"    );
    if ( fabs(corr_sd0)<  cut("SD0",    double()) || ignoreCut("SD0")     ) passCut(ret, "SD0"    );
    if ( nhits         >= cut("NHits",  int()   ) || ignoreCut("NHits")   ) passCut(ret, "NHits"  );
    if ( nValidMuonHits> cut("NValMuHits",int()) || ignoreCut("NValMuHits")) passCut(ret, "NValMuHits"  );
    if ( hcalVeto      <  cut("HCalVeto",double())|| ignoreCut("HCalVeto")) passCut(ret, "HCalVeto");
    if ( ecalVeto      <  cut("ECalVeto",double())|| ignoreCut("ECalVeto")) passCut(ret, "ECalVeto");
    if ( relIso        <  cut("RelIso", double()) || ignoreCut("RelIso")  ) passCut(ret, "RelIso" );

    setIgnored(ret);
    
    return (bool)ret;
  }

  
 private: // member variables
  
  Version_t version_;
  bool recalcDBFromBSp_;
  edm::InputTag beamLineSrc_;
  
};

#endif
