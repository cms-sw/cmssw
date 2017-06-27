#ifndef PhysicsTools_PatUtils_interface_MuonVPlusJetsIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_MuonVPlusJetsIDSelectionFunctor_h

#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

class MuonVPlusJetsIDSelectionFunctor : public Selector<pat::Muon> {

 public: // interface

  bool verbose_;

  enum Version_t { SUMMER08, FIRSTDATA, SPRING10, FALL10, N_VERSIONS, KITQCD };

  MuonVPlusJetsIDSelectionFunctor() {}

#ifndef __GCCXML__
  MuonVPlusJetsIDSelectionFunctor( edm::ParameterSet const & parameters, edm::ConsumesCollector& iC ) :
    MuonVPlusJetsIDSelectionFunctor(parameters)
  {
    beamLineSrcToken_ = iC.consumes<reco::BeamSpot>(beamLineSrc_);
    pvSrcToken_ = iC.consumes<std::vector<reco::Vertex> >(pvSrc_);
  }
#endif

  MuonVPlusJetsIDSelectionFunctor( edm::ParameterSet const & parameters ) {

    verbose_ = false;

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
    else if ( versionStr == "FALL10" ) {
      version = FALL10;
      if (verbose_) std::cout << "\nMUON SELECTION - you are using FALL10 Selection" << std::endl;
    }
    else if (versionStr == "KITQCD") {
      version = KITQCD;
      if (verbose_) std::cout << "\nMUON SELECTION - you are using KITQCD Selection" << std::endl;
    }
    else {
      throw cms::Exception("InvalidInput") << "Expect version to be one of SUMMER08, FIRSTDATA, SPRING10, FALL10" << std::endl;
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
		parameters.getParameter<double>("RelIso"),
		parameters.getParameter<double>("LepZ"),
		parameters.getParameter<int>("nPixelHits"),
		parameters.getParameter<int>("nMatchedStations")
		);
    if ( parameters.exists("cutsToIgnore") )
      setIgnoredCuts( parameters.getParameter<std::vector<std::string> >("cutsToIgnore") );

    retInternal_ = getBitTemplate();

    recalcDBFromBSp_ = parameters.getParameter<bool>("RecalcFromBeamSpot");
    pvSrc_ = parameters.getParameter<edm::InputTag>("pvSrc");
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
				   double reliso = 0.05,
				   double maxLepZ = 1.0,
				   int minPixelHits = 1,
				   int minNMatches = 1
				   ) : recalcDBFromBSp_(false) {
    initialize( version, chi2, d0, ed0, sd0, nhits, nValidMuonHits, ecalveto, hcalveto, reliso,
		maxLepZ, minPixelHits, minNMatches );

    retInternal_ = getBitTemplate();
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
		   double reliso = 0.05,
		   double maxLepZ = 1.0,
		   int minPixelHits = 1,
		   int minNMatches = 1 )
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
    push_back("LepZ",      maxLepZ);
    push_back("nPixelHits",minPixelHits);
    push_back("nMatchedStations", minNMatches);

    set("Chi2");
    set("D0");
    set("ED0");
    set("SD0");
    set("NHits");
    set("NValMuHits");
    set("ECalVeto");
    set("HCalVeto");
    set("RelIso");
    set("LepZ");
    set("nPixelHits");
    set("nMatchedStations");

    indexChi2_          = index_type(&bits_, "Chi2"         );
    indexD0_            = index_type(&bits_, "D0"           );
    indexED0_           = index_type(&bits_, "ED0"          );
    indexSD0_           = index_type(&bits_, "SD0"          );
    indexNHits_         = index_type(&bits_, "NHits"        );
    indexNValMuHits_    = index_type(&bits_, "NValMuHits"   );
    indexECalVeto_      = index_type(&bits_, "ECalVeto"     );
    indexHCalVeto_      = index_type(&bits_, "HCalVeto"     );
    indexRelIso_        = index_type(&bits_, "RelIso"       );
    indexLepZ_          = index_type( &bits_, "LepZ");
    indexPixHits_       = index_type( &bits_, "nPixelHits");
    indexStations_      = index_type( &bits_, "nMatchedStations");

    if ( version == FALL10) {
      set("ED0", false );
      set("SD0", false);
      set("ECalVeto",false);
      set("HCalVeto",false);
    } else if ( version == SPRING10) {
      set("ED0", false );
      set("SD0", false);
      set("ECalVeto",false);
      set("HCalVeto",false);
      set("LepZ", false);
      set("nPixelHits", false);
      set("nMatchedStations", false);
    } else if ( version_ == FIRSTDATA ) {
      set("D0", false );
      set("ED0", false );
      set("NValMuHits",false);
      set("LepZ", false);
      set("nPixelHits", false);
      set("nMatchedStations", false);
    } else if (version == SUMMER08 ) {
      set("SD0", false);
      set("NValMuHits",false);
      set("LepZ", false);
      set("nPixelHits", false);
      set("nMatchedStations", false);

    }

  }

  // Allow for multiple definitions of the cuts.
  bool operator()( const pat::Muon & muon, edm::EventBase const & event, pat::strbitset & ret )
  {

    if (version_ == FALL10 ) return fall10Cuts(muon, event, ret);
    else if (version_ == SPRING10 ) return spring10Cuts(muon, event, ret);
    else if ( version_ == SUMMER08 ) return summer08Cuts( muon, ret );
    else if ( version_ == FIRSTDATA ) return firstDataCuts( muon, ret );
    else if ( version_ == KITQCD ) {
      if (verbose_) std::cout << "Calling KIT selection method" << std::endl;
      return kitQCDCuts (muon, event, ret);
    }
    else {
      return false;
    }
  }

  // For compatibility with older versions.
  bool operator()( const pat::Muon & muon, pat::strbitset & ret )
  {

    if (version_ == SPRING10 || version_ == FALL10 ) throw cms::Exception("LogicError")
      << "MuonVPlusJetsSelectionFunctor SPRING10 and FALL10 versions needs the event! Call operator()(muon,event,ret)"
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

    if ( norm_chi2     <  cut(indexChi2_,   double()) || ignoreCut(indexChi2_)    ) passCut(ret, indexChi2_   );
    if ( fabs(corr_d0) <  cut(indexD0_,     double()) || ignoreCut(indexD0_)      ) passCut(ret, indexD0_     );
    if ( nhits         >= cut(indexNHits_,  int()   ) || ignoreCut(indexNHits_)   ) passCut(ret, indexNHits_  );
    if ( hcalVeto      <  cut(indexHCalVeto_,double())|| ignoreCut(indexHCalVeto_)) passCut(ret, indexHCalVeto_);
    if ( ecalVeto      <  cut(indexECalVeto_,double())|| ignoreCut(indexECalVeto_)) passCut(ret, indexECalVeto_);
    if ( relIso        <  cut(indexRelIso_, double()) || ignoreCut(indexRelIso_)  ) passCut(ret, indexRelIso_ );

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

    if ( norm_chi2     <  cut(indexChi2_,   double()) || ignoreCut(indexChi2_)    ) passCut(ret, indexChi2_   );
    if ( fabs(corr_d0) <  cut(indexD0_,     double()) || ignoreCut(indexD0_)      ) passCut(ret, indexD0_     );
    if ( fabs(corr_ed0)<  cut(indexED0_,    double()) || ignoreCut(indexED0_)     ) passCut(ret, indexED0_    );
    if ( fabs(corr_sd0)<  cut(indexSD0_,    double()) || ignoreCut(indexSD0_)     ) passCut(ret, indexSD0_    );
    if ( nhits         >= cut(indexNHits_,  int()   ) || ignoreCut(indexNHits_)   ) passCut(ret, indexNHits_  );
    if ( hcalVeto      <  cut(indexHCalVeto_,double())|| ignoreCut(indexHCalVeto_)) passCut(ret, indexHCalVeto_);
    if ( ecalVeto      <  cut(indexECalVeto_,double())|| ignoreCut(indexECalVeto_)) passCut(ret, indexECalVeto_);
    if ( relIso        <  cut(indexRelIso_, double()) || ignoreCut(indexRelIso_)  ) passCut(ret, indexRelIso_ );

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

    if ( norm_chi2     <  cut(indexChi2_,   double()) || ignoreCut(indexChi2_)    ) passCut(ret, indexChi2_   );
    if ( fabs(corr_d0) <  cut(indexD0_,     double()) || ignoreCut(indexD0_)      ) passCut(ret, indexD0_     );
    if ( fabs(corr_ed0)<  cut(indexED0_,    double()) || ignoreCut(indexED0_)     ) passCut(ret, indexED0_    );
    if ( fabs(corr_sd0)<  cut(indexSD0_,    double()) || ignoreCut(indexSD0_)     ) passCut(ret, indexSD0_    );
    if ( nhits         >= cut(indexNHits_,  int()   ) || ignoreCut(indexNHits_)   ) passCut(ret, indexNHits_  );
    if ( nValidMuonHits> cut(indexNValMuHits_,int()) || ignoreCut(indexNValMuHits_)) passCut(ret, indexNValMuHits_  );
    if ( hcalVeto      <  cut(indexHCalVeto_,double())|| ignoreCut(indexHCalVeto_)) passCut(ret, indexHCalVeto_);
    if ( ecalVeto      <  cut(indexECalVeto_,double())|| ignoreCut(indexECalVeto_)) passCut(ret, indexECalVeto_);
    if ( relIso        <  cut(indexRelIso_, double()) || ignoreCut(indexRelIso_)  ) passCut(ret, indexRelIso_ );

    setIgnored(ret);

    return (bool)ret;
  }




  // cuts based on top group L+J synchronization exercise
  bool fall10Cuts( const pat::Muon & muon, edm::EventBase const & event, pat::strbitset & ret)
  {

    ret.set(false);

    double norm_chi2 = muon.normChi2();
    double corr_d0 = muon.dB();
    double corr_ed0 = muon.edB();
    double corr_sd0 = ( corr_ed0 > 0.000000001 ) ? corr_d0 / corr_ed0 : 999.0;

    // Get the PV for the muon z requirement
    edm::Handle<std::vector<reco::Vertex> > pvtxHandle_;
    event.getByLabel( pvSrc_, pvtxHandle_ );

    double zvtx = -999;
    if ( pvtxHandle_->size() > 0 ) {
      zvtx = pvtxHandle_->at(0).z();
    } else {
      throw cms::Exception("InvalidInput") << " There needs to be at least one primary vertex in the event." << std::endl;
    }

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


    double z_mu = muon.vertex().z();

    int nPixelHits = muon.innerTrack()->hitPattern().pixelLayersWithMeasurement();

    int nMatchedStations = muon.numberOfMatches();

    if ( norm_chi2     <  cut(indexChi2_,   double()) || ignoreCut(indexChi2_)    ) passCut(ret, indexChi2_   );
    if ( fabs(corr_d0) <  cut(indexD0_,     double()) || ignoreCut(indexD0_)      ) passCut(ret, indexD0_     );
    if ( fabs(corr_ed0)<  cut(indexED0_,    double()) || ignoreCut(indexED0_)     ) passCut(ret, indexED0_    );
    if ( fabs(corr_sd0)<  cut(indexSD0_,    double()) || ignoreCut(indexSD0_)     ) passCut(ret, indexSD0_    );
    if ( nhits         >= cut(indexNHits_,  int()   ) || ignoreCut(indexNHits_)   ) passCut(ret, indexNHits_  );
    if ( nValidMuonHits> cut(indexNValMuHits_,int()) || ignoreCut(indexNValMuHits_)) passCut(ret, indexNValMuHits_  );
    if ( hcalVeto      <  cut(indexHCalVeto_,double())|| ignoreCut(indexHCalVeto_)) passCut(ret, indexHCalVeto_);
    if ( ecalVeto      <  cut(indexECalVeto_,double())|| ignoreCut(indexECalVeto_)) passCut(ret, indexECalVeto_);
    if ( relIso        <  cut(indexRelIso_, double()) || ignoreCut(indexRelIso_)  ) passCut(ret, indexRelIso_ );
    if ( fabs(z_mu-zvtx)<  cut(indexLepZ_, double()) || ignoreCut(indexLepZ_)  ) passCut(ret, indexLepZ_ );
    if ( nPixelHits    >  cut(indexPixHits_,int())    || ignoreCut(indexPixHits_))  passCut(ret, indexPixHits_);
    if ( nMatchedStations> cut(indexStations_,int())    || ignoreCut(indexStations_))  passCut(ret, indexStations_);

    setIgnored(ret);

    return (bool)ret;
  }


  // cuts based on top group L+J synchronization exercise
  // this is a copy of fall 10 cuts
  // with a hack to include a double-sided reliso cut

  bool kitQCDCuts( const pat::Muon & muon, edm::EventBase const & event, pat::strbitset & ret)
  {

    ret.set(false);

    double norm_chi2 = muon.normChi2();
    double corr_d0 = muon.dB();
    double corr_ed0 = muon.edB();
    double corr_sd0 = ( corr_ed0 > 0.000000001 ) ? corr_d0 / corr_ed0 : 999.0;

    // Get the PV for the muon z requirement
    edm::Handle<std::vector<reco::Vertex> > pvtxHandle_;
    event.getByLabel( pvSrc_, pvtxHandle_ );

    double zvtx = -999;
    if ( pvtxHandle_->size() > 0 ) {
      zvtx = pvtxHandle_->at(0).z();
    } else {
      throw cms::Exception("InvalidInput") << " There needs to be at least one primary vertex in the event." << std::endl;
    }

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


    double z_mu = muon.vertex().z();

    int nPixelHits = muon.innerTrack()->hitPattern().pixelLayersWithMeasurement();

    int nMatchedStations = muon.numberOfMatches();

    if ( norm_chi2     <  cut(indexChi2_,   double()) || ignoreCut(indexChi2_)    ) passCut(ret, indexChi2_   );
    if ( fabs(corr_d0) <  cut(indexD0_,     double()) || ignoreCut(indexD0_)      ) passCut(ret, indexD0_     );
    if ( fabs(corr_ed0)<  cut(indexED0_,    double()) || ignoreCut(indexED0_)     ) passCut(ret, indexED0_    );
    if ( fabs(corr_sd0)<  cut(indexSD0_,    double()) || ignoreCut(indexSD0_)     ) passCut(ret, indexSD0_    );
    if ( nhits         >= cut(indexNHits_,  int()   ) || ignoreCut(indexNHits_)   ) passCut(ret, indexNHits_  );
    if ( nValidMuonHits> cut(indexNValMuHits_,int()) || ignoreCut(indexNValMuHits_)) passCut(ret, indexNValMuHits_  );
    if ( hcalVeto      <  cut(indexHCalVeto_,double())|| ignoreCut(indexHCalVeto_)) passCut(ret, indexHCalVeto_);
    if ( ecalVeto      <  cut(indexECalVeto_,double())|| ignoreCut(indexECalVeto_)) passCut(ret, indexECalVeto_);
    if ( fabs(z_mu-zvtx)<  cut(indexLepZ_, double()) || ignoreCut(indexLepZ_)  ) passCut(ret, indexLepZ_ );
    if ( nPixelHits    >  cut(indexPixHits_,int())    || ignoreCut(indexPixHits_))  passCut(ret, indexPixHits_);
    if ( nMatchedStations> cut(indexStations_,int())    || ignoreCut(indexStations_))  passCut(ret, indexStations_);


    ////////////////////////////////////////////////////////////////
    //
    // JMS Dec 13 2010
    // HACK
    // Need double-sided relIso cut to implement data-driven QCD
    //
    //
    ///////////////////////////////////////////////////////////////
    if ( ((relIso > 0.2) && (relIso < 0.75))
         || ignoreCut(indexRelIso_)  ) passCut(ret, indexRelIso_ );




    setIgnored(ret);

    return (bool)ret;
  }




 private: // member variables

  Version_t version_;
  bool recalcDBFromBSp_;
  edm::InputTag beamLineSrc_;
#ifndef __GCCXML__
  edm::EDGetTokenT<reco::BeamSpot> beamLineSrcToken_;
#endif
  edm::InputTag pvSrc_;
#ifndef __GCCXML__
  edm::EDGetTokenT<std::vector<reco::Vertex> > pvSrcToken_;
#endif

  index_type indexChi2_;
  index_type indexD0_;
  index_type indexED0_;
  index_type indexSD0_;
  index_type indexNHits_;
  index_type indexNValMuHits_;
  index_type indexECalVeto_;
  index_type indexHCalVeto_;
  index_type indexRelIso_;
  index_type indexLepZ_;
  index_type indexPixHits_;
  index_type indexStations_;


};

#endif
