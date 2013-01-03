//Alexander.Schmidt@cern.ch
//March 2007

// This code was created during the debugging of the fast RecHits and fast 
// tracks.  It produces some validation and debugging plots of the RecHits and tracks.

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FastSimulation/Tracking/test/FastTrackAnalyzer.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h" 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

// Numbering scheme
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TH1F.h"
#include "TFile.h"

using namespace edm;
using namespace std;
    
//---------------------------------------------------------
FastTrackAnalyzer::FastTrackAnalyzer(edm::ParameterSet const& conf) : 
  conf_(conf),
  // This is very likely what you want in the configuration for the
  // following two parameters (I would put this in the cfi file instead
  // of this comment if there was one)
  //  simVertexContainerTag = cms.InputTag('famosSimHits'),
  //  siTrackerGSRecHit2DCollectionTag = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSRecHits")
  simVertexContainerTag(conf.getParameter<edm::InputTag>("simVertexContainerTag")),
  siTrackerGSRecHit2DCollectionTag(conf.getParameter<edm::InputTag>("siTrackerGSRecHit2DCollectionTag")) {
  
  iEventCounter=0;
  
  trackProducer = conf_.getParameter<std::string>("TrackProducer");

  trackerContainers.clear();
  trackerContainers = conf.getParameter<std::vector<std::string> >("SimHitList");

  // histogram axis limits for RecHit validation plots
  PXB_Res_AxisLim = conf.getParameter<double>("PXB_Res_AxisLim" );  
  PXF_Res_AxisLim = conf.getParameter<double>("PXF_Res_AxisLim" );
  PXB_RecPos_AxisLim = conf.getParameter<double>("PXB_RecPos_AxisLim" );
  PXF_RecPos_AxisLim = conf.getParameter<double>("PXF_RecPos_AxisLim" );
  PXB_SimPos_AxisLim = conf.getParameter<double>("PXB_SimPos_AxisLim" );
  PXF_SimPos_AxisLim = conf.getParameter<double>("PXF_SimPos_AxisLim" );
  PXB_Err_AxisLim = conf.getParameter<double>("PXB_Err_AxisLim");
  PXF_Err_AxisLim = conf.getParameter<double>("PXF_Err_AxisLim");
  
  TIB_Res_AxisLim = conf.getParameter<double>("TIB_Res_AxisLim" );
  TIB_Resy_AxisLim = conf.getParameter<double>("TIB_Resy_AxisLim" );
  TIB_Pos_AxisLim = conf.getParameter<double>("TIB_Pos_AxisLim" );
  TID_Res_AxisLim = conf.getParameter<double>("TID_Res_AxisLim" );
  TID_Resy_AxisLim = conf.getParameter<double>("TID_Resy_AxisLim" );
  TID_Pos_AxisLim = conf.getParameter<double>("TID_Pos_AxisLim" );
  TOB_Res_AxisLim = conf.getParameter<double>("TOB_Res_AxisLim" );
  TOB_Resy_AxisLim = conf.getParameter<double>("TOB_Resy_AxisLim" );
  TOB_Pos_AxisLim = conf.getParameter<double>("TOB_Pos_AxisLim" );
  TEC_Res_AxisLim = conf.getParameter<double>("TEC_Res_AxisLim" );
  TEC_Resy_AxisLim = conf.getParameter<double>("TEC_Resy_AxisLim" );
  TEC_Pos_AxisLim = conf.getParameter<double>("TEC_Pos_AxisLim" );
  
  TIB_Err_AxisLim  = conf.getParameter<double>("TIB_Err_AxisLim" );
  TIB_Erry_AxisLim  = conf.getParameter<double>("TIB_Erry_AxisLim" );
  TID_Err_AxisLim  = conf.getParameter<double>("TID_Err_AxisLim" );
  TID_Erry_AxisLim  = conf.getParameter<double>("TID_Erry_AxisLim" );
  TOB_Err_AxisLim  = conf.getParameter<double>("TOB_Err_AxisLim" );
  TOB_Erry_AxisLim  = conf.getParameter<double>("TOB_Erry_AxisLim" );
  TEC_Err_AxisLim  = conf.getParameter<double>("TEC_Err_AxisLim" );
  TEC_Erry_AxisLim  = conf.getParameter<double>("TEC_Erry_AxisLim" );

  NumTracks_AxisLim = conf.getParameter<int>("NumTracks_AxisLim" );
 outfilename = conf.getParameter<string>("outfilename");
}
//---------------------------------------------------------
FastTrackAnalyzer::~FastTrackAnalyzer() {}
//---------------------------------------------------------
void FastTrackAnalyzer::beginRun(edm::Run const& run, edm::EventSetup const& es){
  
  es.get<IdealMagneticFieldRecord>().get(theMagField);
  
  // book histograms  for plots

  // number of pixel hits
  hMap["all_NumSimPixHits"] = new TH1F("all_NumSimPixHits", "all_NumSimPixHits", 100, 0,10);
  hMap["all_NumRecPixHits"] = new TH1F("all_NumRecPixHits", "all_NumRecPixHits", 100, 0,10);

  //number of strip hits
  hMap["all_NumSimStripHits"] = new TH1F("all_NumSimStripHits", "all_NumSimStripHits", 100, 0,30);
  hMap["all_NumRecStripHits"] = new TH1F("all_NumRecStripHits", "all_NumRecStripHits", 100, 0,30);

  // number of total hits
  hMap["all_NumSimHits"] = new TH1F("all_NumSimHits", "all_NumSimHits", 100, 0,35);
  hMap["all_NumRecHits"] = new TH1F("all_NumRecHits", "all_NumRecHits", 100, 0,35);

  // number of total hits in track candidates
  hMap["cnd_NumRecHits"] = new TH1F("cnd_NumRecHits", "cnd_NumRecHits", 100, 0,35);

  // global position of all hits
  hMap["all_GlobSimPos_x"] = new TH1F("all_GlobSimPos_x","all_GlobSimPos_x" ,300, 0,115);
  hMap["all_GlobSimPos_y"] = new TH1F("all_GlobSimPos_y","all_GlobSimPos_y" ,300, 0,115);
  hMap["all_GlobSimPos_z"] = new TH1F("all_GlobSimPos_z","all_GlobSimPos_z" ,300, 0,300);

  hMap["all_GlobRecPos_x"] = new TH1F("all_GlobRecPos_x","all_GlobRecPos_x" ,300, 0,115);
  hMap["all_GlobRecPos_y"] = new TH1F("all_GlobRecPos_y","all_GlobRecPos_y" ,300, 0,115);
  hMap["all_GlobRecPos_z"] = new TH1F("all_GlobRecPos_z","all_GlobRecPos_z" ,300, 0,300);

  // pixel maximum 3 layers
  for(int i=1; i<=3; i++){
    TString index = ""; index+=i;
    hMap["all_PXB_Res_x_"+index] = new TH1F("all_PXB_Res_x_"+index, "all_PXB_Res_x_"+index, 100, - PXB_Res_AxisLim,  PXB_Res_AxisLim);
    hMap["all_PXB_Res_y_"+index] = new TH1F("all_PXB_Res_y_"+index, "all_PXB_Res_y_"+index, 100, - PXB_Res_AxisLim,  PXB_Res_AxisLim);
    hMap["all_PXB_SimPos_x_"+index] = new TH1F("all_PXB_SimPos_x_"+index, "all_PXB_SimPos_x_"+index, 100, -PXB_SimPos_AxisLim, PXB_SimPos_AxisLim);
    hMap["all_PXB_SimPos_y_"+index] = new TH1F("all_PXB_SimPos_y_"+index, "all_PXB_SimPos_y_"+index, 100, -PXB_SimPos_AxisLim, PXB_SimPos_AxisLim);
    hMap["all_PXB_RecPos_x_"+index] = new TH1F("all_PXB_RecPos_x_"+index, "all_PXB_RecPos_x_"+index, 100, -PXB_RecPos_AxisLim, PXB_RecPos_AxisLim);
    hMap["all_PXB_RecPos_y_"+index] = new TH1F("all_PXB_RecPos_y_"+index, "all_PXB_RecPos_y_"+index, 100, -PXB_RecPos_AxisLim, PXB_RecPos_AxisLim);

    hMap["all_PXB_Err_xx"+index] = new TH1F("all_PXB_Err_xx"+index, "all_PXB_Err_xx"+index, 100,  -PXB_Err_AxisLim, PXB_Err_AxisLim);
    hMap["all_PXB_Err_xy"+index] = new TH1F("all_PXB_Err_xy"+index, "all_PXB_Err_xy"+index, 100,  -PXB_Err_AxisLim, PXB_Err_AxisLim);
    hMap["all_PXB_Err_yy"+index] = new TH1F("all_PXB_Err_yy"+index, "all_PXB_Err_yy"+index, 100,  -PXB_Err_AxisLim, PXB_Err_AxisLim);


    hMap["PXB_Res_x_"+index] = new TH1F("PXB_Res_x_"+index, "PXB_Res_x_"+index, 100, - PXB_Res_AxisLim,  PXB_Res_AxisLim);
    hMap["PXB_Res_y_"+index] = new TH1F("PXB_Res_y_"+index, "PXB_Res_y_"+index, 100, - PXB_Res_AxisLim,  PXB_Res_AxisLim);
    hMap["PXB_SimPos_x_"+index] = new TH1F("PXB_SimPos_x_"+index, "PXB_SimPos_x_"+index, 100, -PXB_SimPos_AxisLim, PXB_SimPos_AxisLim);
    hMap["PXB_SimPos_y_"+index] = new TH1F("PXB_SimPos_y_"+index, "PXB_SimPos_y_"+index, 100, -PXB_SimPos_AxisLim, PXB_SimPos_AxisLim);
    hMap["PXB_RecPos_x_"+index] = new TH1F("PXB_RecPos_x_"+index, "PXB_RecPos_x_"+index, 100, -PXB_RecPos_AxisLim, PXB_RecPos_AxisLim);
    hMap["PXB_RecPos_y_"+index] = new TH1F("PXB_RecPos_y_"+index, "PXB_RecPos_y_"+index, 100, -PXB_RecPos_AxisLim, PXB_RecPos_AxisLim);
    hMap["PXB_Err_xx"+index] = new TH1F("PXB_Err_xx"+index, "PXB_Err_xx"+index, 100,  -PXB_Err_AxisLim, PXB_Err_AxisLim);
    hMap["PXB_Err_xy"+index] = new TH1F("PXB_Err_xy"+index, "PXB_Err_xy"+index, 100,  -PXB_Err_AxisLim, PXB_Err_AxisLim);
    hMap["PXB_Err_yy"+index] = new TH1F("PXB_Err_yy"+index, "PXB_Err_yy"+index, 100,  -PXB_Err_AxisLim, PXB_Err_AxisLim);


  }
  for(int i=1;i<=2;i++){
    TString index = ""; index+=i;
    hMap["all_PXF_Res_x_"+index] = new TH1F("all_PXF_Res_x_"+index, "all_PXF_Res_x_"+index, 100, -PXF_Res_AxisLim, PXF_Res_AxisLim);
    hMap["all_PXF_Res_y_"+index] = new TH1F("all_PXF_Res_y_"+index, "all_PXF_Res_y_"+index, 100, -PXF_Res_AxisLim, PXF_Res_AxisLim);
    hMap["all_PXF_SimPos_x_"+index] = new TH1F("all_PXF_SimPos_x_"+index, "all_PXF_SimPos_x_"+index, 100, -PXF_SimPos_AxisLim, PXF_SimPos_AxisLim);
    hMap["all_PXF_SimPos_y_"+index] = new TH1F("all_PXF_SimPos_y_"+index, "all_PXF_SimPos_y_"+index, 100, -PXF_SimPos_AxisLim, PXF_SimPos_AxisLim);
    hMap["all_PXF_RecPos_x_"+index] = new TH1F("all_PXF_RecPos_x_"+index, "all_PXF_RecPos_x_"+index, 100, -PXF_RecPos_AxisLim, PXF_RecPos_AxisLim);
    hMap["all_PXF_RecPos_y_"+index] = new TH1F("all_PXF_RecPos_y_"+index, "all_PXF_RecPos_y_"+index, 100, -PXF_RecPos_AxisLim, PXF_RecPos_AxisLim);
    hMap["all_PXF_Err_xx"+index] = new TH1F("all_PXF_Err_xx"+index, "all_PXF_Err_xx"+index, 100,  -PXF_Err_AxisLim, PXF_Err_AxisLim);
    hMap["all_PXF_Err_xy"+index] = new TH1F("all_PXF_Err_xy"+index, "all_PXF_Err_xy"+index, 100,  -PXF_Err_AxisLim, PXF_Err_AxisLim);
    hMap["all_PXF_Err_yy"+index] = new TH1F("all_PXF_Err_yy"+index, "all_PXF_Err_yy"+index, 100,  -PXF_Err_AxisLim, PXF_Err_AxisLim);

    hMap["PXF_Res_x_"+index] = new TH1F("PXF_Res_x_"+index, "PXF_Res_x_"+index, 100, -PXF_Res_AxisLim, PXF_Res_AxisLim);
    hMap["PXF_Res_y_"+index] = new TH1F("PXF_Res_y_"+index, "PXF_Res_y_"+index, 100, -PXF_Res_AxisLim, PXF_Res_AxisLim);
    hMap["PXF_SimPos_x_"+index] = new TH1F("PXF_SimPos_x_"+index, "PXF_SimPos_x_"+index, 100, -PXF_SimPos_AxisLim, PXF_SimPos_AxisLim);
    hMap["PXF_SimPos_y_"+index] = new TH1F("PXF_SimPos_y_"+index, "PXF_SimPos_y_"+index, 100, -PXF_SimPos_AxisLim, PXF_SimPos_AxisLim);
    hMap["PXF_RecPos_x_"+index] = new TH1F("PXF_RecPos_x_"+index, "PXF_RecPos_x_"+index, 100, -PXF_RecPos_AxisLim, PXF_RecPos_AxisLim);
    hMap["PXF_RecPos_y_"+index] = new TH1F("PXF_RecPos_y_"+index, "PXF_RecPos_y_"+index, 100, -PXF_RecPos_AxisLim, PXF_RecPos_AxisLim);
    hMap["PXF_Err_xx"+index] = new TH1F("PXF_Err_xx"+index, "PXF_Err_xx"+index, 100,  -PXF_Err_AxisLim, PXF_Err_AxisLim);
    hMap["PXF_Err_xy"+index] = new TH1F("PXF_Err_xy"+index, "PXF_Err_xy"+index, 100,  -PXF_Err_AxisLim, PXF_Err_AxisLim);
    hMap["PXF_Err_yy"+index] = new TH1F("PXF_Err_yy"+index, "PXF_Err_yy"+index, 100,  -PXF_Err_AxisLim, PXF_Err_AxisLim);

  }

  // strip maximum 7 layers (TEC)
  for(int i=1; i<=7; i++){
    TString index = ""; index+=i;
    if(i<5){
      hMap["all_TIB_Res_x_"+index] = new TH1F("all_TIB_Res_x_"+index, "all_TIB_Res_x_"+index, 100, -TIB_Res_AxisLim, TIB_Res_AxisLim);
      hMap["all_TIB_SimPos_x_"+index] = new TH1F("all_TIB_SimPos_x_"+index, "all_TIB_SimPos_x_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
      hMap["all_TIB_RecPos_x_"+index] = new TH1F("all_TIB_RecPos_x_"+index, "all_TIB_RecPos_x_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
      hMap["all_TIB_Err_x_"+index] = new TH1F("all_TIB_Err_x_"+index, "all_TIB_Err_x_"+index, 100, 0, TIB_Err_AxisLim);
      hMap["TIB_Res_x_"+index] = new TH1F("TIB_Res_x_"+index, "TIB_Res_x_"+index, 100, -TIB_Res_AxisLim, TIB_Res_AxisLim);
      hMap["TIB_SimPos_x_"+index] = new TH1F("TIB_SimPos_x_"+index, "TIB_SimPos_x_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
      hMap["TIB_RecPos_x_"+index] = new TH1F("TIB_RecPos_x_"+index, "TIB_RecPos_x_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
      hMap["TIB_Err_x_"+index] = new TH1F("TIB_Err_x_"+index, "TIB_Err_x_"+index, 100, 0, TIB_Err_AxisLim);
      //pat
      if(i==1 || i== 2){
	hMap["all_TIB_Res_y_"+index] = new TH1F("all_TIB_Res_y_"+index, "all_TIB_Res_y_"+index, 100, -TIB_Resy_AxisLim, TIB_Resy_AxisLim);
	hMap["all_TIB_RecPos_y_"+index] = new TH1F("all_TIB_RecPos_y_"+index, "all_TIB_RecPos_y_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
	hMap["all_TIB_Err_y_"+index] = new TH1F("all_TIB_Err_y_"+index, "all_TIB_Err_y_"+index, 100, 0, TIB_Erry_AxisLim);
	hMap["all_TIB_SimPos_y_"+index] = new TH1F("all_TIB_SimPos_y_"+index, "all_TIB_SimPos_y_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
	hMap["TIB_Res_y_"+index] = new TH1F("TIB_Res_y_"+index, "TIB_Res_y_"+index, 100, -TIB_Resy_AxisLim, TIB_Resy_AxisLim);
	hMap["TIB_RecPos_y_"+index] = new TH1F("TIB_RecPos_y_"+index, "TIB_RecPos_y_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
	hMap["TIB_Err_y_"+index] = new TH1F("TIB_Err_y_"+index, "TIB_Err_y_"+index, 100, 0, TIB_Erry_AxisLim);
	hMap["TIB_SimPos_y_"+index] = new TH1F("TIB_SimPos_y_"+index, "TIB_SimPos_y_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
      }
    }
    if(i<7){
      hMap["all_TOB_Res_x_"+index] = new TH1F("all_TOB_Res_x_"+index, "all_TOB_Res_x_"+index, 100, -TOB_Res_AxisLim, TOB_Res_AxisLim);
      hMap["all_TOB_SimPos_x_"+index] = new TH1F("all_TOB_SimPos_x_"+index, "all_TOB_SimPos_x_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
      hMap["all_TOB_RecPos_x_"+index] = new TH1F("all_TOB_RecPos_x_"+index, "all_TOB_RecPos_x_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
      hMap["all_TOB_Err_x_"+index] = new TH1F("all_TOB_Err_x_"+index, "all_TOB_Err_x_"+index, 100, 0, TOB_Err_AxisLim);
      hMap["TOB_Res_x_"+index] = new TH1F("TOB_Res_x_"+index, "TOB_Res_x_"+index, 100, -TOB_Res_AxisLim, TOB_Res_AxisLim);
      hMap["TOB_SimPos_x_"+index] = new TH1F("TOB_SimPos_x_"+index, "TOB_SimPos_x_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
      hMap["TOB_RecPos_x_"+index] = new TH1F("TOB_RecPos_x_"+index, "TOB_RecPos_x_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
      hMap["TOB_Err_x_"+index] = new TH1F("TOB_Err_x_"+index, "TOB_Err_x_"+index, 100, -TOB_Err_AxisLim, TOB_Err_AxisLim);
      //pat
      if(i==1 || i== 2){
	hMap["all_TOB_Res_y_"+index] = new TH1F("all_TOB_Res_y_"+index, "all_TOB_Res_y_"+index, 100, -TOB_Resy_AxisLim, TOB_Resy_AxisLim);
	hMap["all_TOB_RecPos_y_"+index] = new TH1F("all_TOB_RecPos_y_"+index, "all_TOB_RecPos_y_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
	hMap["all_TOB_Err_y_"+index] = new TH1F("all_TOB_Err_y_"+index, "all_TOB_Err_y_"+index, 100, 0, TOB_Erry_AxisLim);
	hMap["all_TOB_SimPos_y_"+index] = new TH1F("all_TOB_SimPos_y_"+index, "all_TOB_SimPos_y_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
	hMap["TOB_Res_y_"+index] = new TH1F("TOB_Res_y_"+index, "TOB_Res_y_"+index, 100, -TOB_Resy_AxisLim, TOB_Resy_AxisLim);
	hMap["TOB_RecPos_y_"+index] = new TH1F("TOB_RecPos_y_"+index, "TOB_RecPos_y_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
	hMap["TOB_Err_y_"+index] = new TH1F("TOB_Err_y_"+index, "TOB_Err_y_"+index, 100, 0, TOB_Erry_AxisLim);
	hMap["TOB_SimPos_y_"+index] = new TH1F("TOB_SimPos_y_"+index, "TOB_SimPos_y_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
      }
    }

    hMap["all_TEC_Res_x_"+index] = new TH1F("all_TEC_Res_x_"+index, "all_TEC_Res_x_"+index, 100, -TEC_Res_AxisLim, TEC_Res_AxisLim);
    hMap["all_TEC_Res_x_proj_"+index] = new TH1F("all_TEC_Res_x_proj_"+index, "all_TEC_Res_x_proj_"+index, 100, -TEC_Res_AxisLim, TEC_Res_AxisLim);
    hMap["all_TEC_SimPos_x_"+index] = new TH1F("all_TEC_SimPos_x_"+index, "all_TEC_SimPos_x_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
    hMap["all_TEC_RecPos_x_"+index] = new TH1F("all_TEC_RecPos_x_"+index, "all_TEC_RecPos_x_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
    hMap["all_TEC_Err_x_"+index] = new TH1F("all_TEC_Err_x_"+index, "all_TEC_Err_x_"+index, 100, 0, TEC_Err_AxisLim);


    hMap["TEC_Res_x_"+index] = new TH1F("TEC_Res_x_"+index, "TEC_Res_x_"+index, 100, -TEC_Res_AxisLim, TEC_Res_AxisLim);
    hMap["TEC_Res_x_proj_"+index] = new TH1F("TEC_Res_x_proj_"+index, "TEC_Res_x_proj_"+index, 100, -TEC_Res_AxisLim, TEC_Res_AxisLim);
    hMap["TEC_SimPos_x_"+index] = new TH1F("TEC_SimPos_x_"+index, "TEC_SimPos_x_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
    hMap["TEC_RecPos_x_"+index] = new TH1F("TEC_RecPos_x_"+index, "TEC_RecPos_x_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
    hMap["TEC_Err_x_"+index] = new TH1F("TEC_Err_x_"+index, "TEC_Err_x_"+index, 100, -TEC_Err_AxisLim, TEC_Err_AxisLim);
      //pat
      if(i==1 || i== 2 || i==5){
	hMap["all_TEC_Res_y_"+index] = new TH1F("all_TEC_Res_y_"+index, "all_TEC_Res_y_"+index, 100, -TEC_Resy_AxisLim, TEC_Resy_AxisLim);
	hMap["all_TEC_RecPos_y_"+index] = new TH1F("all_TEC_RecPos_y_"+index, "all_TEC_RecPos_y_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
	hMap["all_TEC_Err_y_"+index] = new TH1F("all_TEC_Err_y_"+index, "all_TEC_Err_y_"+index, 100,0, TEC_Erry_AxisLim);
	hMap["all_TEC_SimPos_y_"+index] = new TH1F("all_TEC_SimPos_y_"+index, "all_TEC_SimPos_y_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
	hMap["TEC_Res_y_"+index] = new TH1F("TEC_Res_y_"+index, "TEC_Res_y_"+index, 100, -TEC_Resy_AxisLim, TEC_Resy_AxisLim);
	hMap["TEC_RecPos_y_"+index] = new TH1F("TEC_RecPos_y_"+index, "TEC_RecPos_y_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
	hMap["TEC_Err_y_"+index] = new TH1F("TEC_Err_y_"+index, "TEC_Err_y_"+index, 100,0, TEC_Erry_AxisLim);
	hMap["TEC_SimPos_y_"+index] = new TH1F("TEC_SimPos_y_"+index, "TEC_SimPos_y_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
      }

    if(i<4){
      hMap["all_TID_Res_x_"+index] = new TH1F("all_TID_Res_x_"+index, "all_TID_Res_x_"+index, 100, -TID_Res_AxisLim, TID_Res_AxisLim);
      hMap["all_TID_Res_x_proj_"+index] = new TH1F("all_TID_Res_x_proj_"+index, "all_TID_Res_x_proj_"+index, 100, -TID_Res_AxisLim, TID_Res_AxisLim);
      hMap["all_TID_SimPos_x_"+index] = new TH1F("all_TID_SimPos_x_"+index, "all_TID_SimPos_x_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
      hMap["all_TID_RecPos_x_"+index] = new TH1F("all_TID_RecPos_x_"+index, "all_TID_RecPos_x_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
      hMap["all_TID_Err_x_"+index] = new TH1F("all_TID_Err_x_"+index, "all_TID_Err_x_"+index, 100, 0, TID_Err_AxisLim);
      hMap["TID_Res_x_"+index] = new TH1F("TID_Res_x_"+index, "TID_Res_x_"+index, 100, -TID_Res_AxisLim, TID_Res_AxisLim);
      hMap["TID_Res_x_proj_"+index] = new TH1F("TID_Res_x_proj_"+index, "TID_Res_x_proj_"+index, 100, -TID_Res_AxisLim, TID_Res_AxisLim);
      hMap["TID_SimPos_x_"+index] = new TH1F("TID_SimPos_x_"+index, "TID_SimPos_x_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
      hMap["TID_RecPos_x_"+index] = new TH1F("TID_RecPos_x_"+index, "TID_RecPos_x_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
      hMap["TID_Err_x_"+index] = new TH1F("TID_Err_x_"+index, "TID_Err_x_"+index, 100, -TID_Err_AxisLim, TID_Err_AxisLim);
      //pat
      if(i==1 || i== 2){
	hMap["all_TID_Res_y_"+index] = new TH1F("all_TID_Res_y_"+index, "all_TID_Res_y_"+index, 100, -TID_Resy_AxisLim, TID_Resy_AxisLim);
	hMap["all_TID_RecPos_y_"+index] = new TH1F("all_TID_RecPos_y_"+index, "all_TID_RecPos_y_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
	hMap["all_TID_Err_y_"+index] = new TH1F("all_TID_Err_y_"+index, "all_TID_Err_y_"+index, 100,0, TID_Erry_AxisLim);
	hMap["all_TID_SimPos_y_"+index] = new TH1F("all_TID_SimPos_y_"+index, "all_TID_SimPos_y_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
	hMap["TID_Res_y_"+index] = new TH1F("TID_Res_y_"+index, "TID_Res_y_"+index, 100, -TID_Resy_AxisLim, TID_Resy_AxisLim);
	hMap["TID_RecPos_y_"+index] = new TH1F("TID_RecPos_y_"+index, "TID_RecPos_y_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
	hMap["TID_Err_y_"+index] = new TH1F("TID_Err_y_"+index, "TID_Err_y_"+index, 100,0, TID_Erry_AxisLim);
	hMap["TID_SimPos_y_"+index] = new TH1F("TID_SimPos_y_"+index, "TID_SimPos_y_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);

      }
    }
  }

  // plots for tracks
  hMap["NumRecTracks"] = new TH1F("NumRecTracks", "NumRecTracks", 100, 0, NumTracks_AxisLim);
  hMap["NumSimTracks"] = new TH1F("NumSimTracks", "NumSimTracks", 100, 0, NumTracks_AxisLim);
 
  hMap["trk_Rec_phi"] = new TH1F("trk_Rec_phi", "trk_Rec_phi", 100,-3.5, 3.5);
  hMap["trk_Sim_phi"] = new TH1F("trk_Sim_phi", "trk_Sim_phi", 100,-3.5, 3.5);
  hMap["trk_Res_phi"] = new TH1F("trk_Res_phi", "trk_Res_phi", 1000,-0.02, 0.02);
  hMap["trk_Pull_phi"] = new TH1F("trk_Pull_phi", "trk_Pull_phi", 100,-15, 15);

  hMap["trk_Rec_eta"] = new TH1F("trk_Rec_eta", "trk_Rec_eta", 100,-3, 3);
  hMap["trk_Sim_eta"] = new TH1F("trk_Sim_eta", "trk_Sim_eta", 100,-3, 3);
  hMap["trk_Cnd_eta"] = new TH1F("trk_Cnd_eta", "trk_Cnd_eta", 100,-3, 3);
  hMap["trk_Res_eta"] = new TH1F("trk_Res_eta", "trk_Res_eta", 1000,-0.01, 0.01);
  hMap["trk_Pull_eta"] = new TH1F("trk_Pull_eta", "trk_Pull_eta", 1000,-25, 25);

  hMap["trk_Rec_pt"] = new TH1F("trk_Rec_pt", "trk_Rec_pt", 100,0, 100);
  hMap["trk_Sim_pt"] = new TH1F("trk_Sim_pt", "trk_Sim_pt", 100,0, 100);
  hMap["trk_Res_pt"] = new TH1F("trk_Res_pt", "trk_Res_pt", 100,-10, 10);
  hMap["trk_Pull_pt"] = new TH1F("trk_Pull_pt", "trk_Pull_pt", 100,-10, 10);

  hMap["trk_Pull_qoverp"] = new TH1F("trk_Pull_qoverp", "trk_Pull_qoverp", 100,-25, 25);
  hMap["trk_Rec_qoverp"] = new TH1F("trk_Rec_qoverp", "trk_Rec_qoverp", 100,-25, 25);

  hMap["trk_Rec_chi2"] = new TH1F("trk_Rec_chi2", "trk_Rec_chi2", 100,0, 100);
  hMap["trk_Rec_Normchi2"] = new TH1F("trk_Rec_Normchi2", "trk_Rec_Normchi2", 100,0, 12);
  hMap["trk_Rec_ndof"] = new TH1F("trk_Rec_ndof", "trk_Rec_ndof", 100,0, 100);

  hMap["trk_Res_d0"] = new TH1F("trk_Res_d0", "trk_Res_d0", 100,-0.1, 0.1);
  hMap["trk_Rec_d0"] = new TH1F("trk_Rec_d0", "trk_Rec_d0", 100,-0.02, 0.02);
  hMap["trk_Err_d0"] = new TH1F("trk_Err_d0", "trk_Err_d0", 100,0, 0.004);
  hMap["trk_Pull_d0"] = new TH1F("trk_Pull_d0", "trk_Pull_d0", 100,-10, 10);

  hMap["trk_Res_dz"] = new TH1F("trk_Res_dz", "trk_Res_dz", 100,-0.1, 0.1);
  hMap["trk_Rec_dz"] = new TH1F("trk_Rec_dz", "trk_Rec_dz", 100,-10, 10);
  hMap["trk_Err_dz"] = new TH1F("trk_Err_dz", "trk_Err_dz", 100,0, 0.02);
  hMap["trk_Pull_dz"] = new TH1F("trk_Pull_dz", "trk_Pull_dz", 100,-10, 10);

  hMap["trk_Rec_charge"] = new TH1F("trk_Rec_charge", "trk_Rec_charge", 100, -2,2);

}
//---------------------------------------------------------
void FastTrackAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
  {
    iEventCounter++;

    // get the geometry
    edm::ESHandle<TrackerGeometry> theG;
    setup.get<TrackerDigiGeometryRecord>().get(theG);
    const TrackerGeometry &tracker(*theG);
    trackerG = &tracker;

    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHand;
    setup.get<IdealGeometryRecord>().get(tTopoHand);
    const TrackerTopology *tTopo=tTopoHand.product();

    std::cout << "\nEvent ID = "<< event.id() << std::endl ;
    
    // get rec track collection
    edm::Handle<reco::TrackCollection> trackCollection;
    event.getByLabel(trackProducer, trackCollection);
    const reco::TrackCollection tC = *(trackCollection.product());

    
    //get simtrack info
     edm::Handle<std::vector<SimTrack> > theSimTracks;
     event.getByLabel("famosSimHits",theSimTracks); 

    edm::Handle<SimVertexContainer> theSimVtx;
    event.getByLabel(simVertexContainerTag, theSimVtx);

    // print size of vertex collection
    std::cout<<" AS: vertex.size() = "<< theSimVtx->size() << std::endl;

    std::vector<unsigned int> SimTrackIds;


    // Get PSimHit's of the Event
    edm::Handle<CrossingFrame<PSimHit> > cf_simhit; 
    std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
    for(uint32_t i=0; i<trackerContainers.size(); i++){
      event.getByLabel("mix",trackerContainers[i], cf_simhit);
      cf_simhitvec.push_back(cf_simhit.product());
    }
    std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
    
    //Get RecHits from the event
    edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
    event.getByLabel(siTrackerGSRecHit2DCollectionTag, theGSRecHits);
    // stop with error if empty RecHit collection
    if(theGSRecHits->size() == 0) {
      std::cout<<" AS: theGSRecHits->size() == 0" << std::endl;
      exit(1);
    }


    //------------------------------------------
    // validation plots for all hits
    edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator theRecHitIteratorBegin = theGSRecHits->begin();
    edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator theRecHitIteratorEnd   = theGSRecHits->end();
    
    //counters for number of hits
    unsigned int iPixSimHits = 0;
    unsigned int iPixRecHits = 0;
    unsigned int iStripSimHits=0;
    unsigned int iStripRecHits=0;
    unsigned int iTotalRecHits = 0;
    unsigned int iTotalSimHits = 0;
    unsigned int iCandRecHits = 0;
    
    
    //------------------------
    //loop on all rechits, match them to simhits and make plots
    for(edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator iterrechit = theRecHitIteratorBegin;
	iterrechit != theRecHitIteratorEnd; ++iterrechit) { // loop on GSRecHits
      
      //      std::cout<<" looping over rechits "<< std::endl;

      DetId detid = (*iterrechit).geographicalId();

      // count rechits
      unsigned int subdet = detid.subdetId();
      //      std::cout<<" subdet = "<< subdet << std::endl;
      //      std::cout<<" recdetid = " << detid.rawId() << std::endl;
      iTotalRecHits++;
      if(subdet==1 || subdet==2) iPixRecHits++;
      else if(subdet==3|| subdet==4 || subdet==5 || subdet == 6) iStripRecHits++;
      else { // for debugging
	std::cout<<" AS: ERROR simhit subdet inconsistent"<< std::endl;
	exit(1);
      }

//       //continue for matched hits
//       if(subdet > 2){
// 	StripSubdetector specDetId=StripSubdetector(detid);
// 	if(specDetId.glued()) continue;
//       }

      SiTrackerGSRecHit2D const rechit=*iterrechit;

      //match sim hit to rec hit
      unsigned int matchedSimHits = 0;
      const PSimHit* simHit = NULL;
      const PSimHit* simHitStereoPatch = NULL;
      int numpartners=0;
      for (MixCollection<PSimHit>::iterator isim=(*allTrackerHits).begin(); isim!= (*allTrackerHits).end(); isim++) {
	//	std::cout<<" looping over simhits " << std::endl;
	//compare detUnitIds && SimTrackIds to match rechit to simhit (for pileup will need to add EncodedEventId info as well).
	// int simdetid = (*isim).detUnitId();
	//	std::cout<<"  simdetid = "<< simdetid << std::endl;
	if((int) detid.rawId() == (int) (*isim).detUnitId() && 
	   (int) (*iterrechit).simtrackId()== (int) (*isim).trackId())
	  {
	    matchedSimHits++;
	    numpartners++;
	    simHit = &*isim;
	    
	    /*
	    std::cout << "\tRecHit pos = " << rechit.localPosition() << "\tin Det " << detid.rawId() << "\tsimtkID = " << (*iterrechit).simtrackId() << std::endl;
	    std::cout << "\tmatched to Simhit = " << (*isim).localPosition() << "\tsimtkId = " <<(*isim).trackId() << std::endl; 	    
	    */
	  }
	else{
	  if(subdet > 2){
	    //	    StripSubdetector specDetId=StripSubdetector(detid);
	    //	    std::cout<<"   glued = " << specDetId.glued() << std::endl;
	    const GluedGeomDet* gluedDet = dynamic_cast<const GluedGeomDet*> (trackerG->idToDet(detid));
	    //	    std::cout<<"   gluedgeomdet = " << gluedDet << std::endl;
	    if(gluedDet){
	      //	      const GluedGeomDet* gluedDet = (const GluedGeomDet*)trackerG->idToDet(detid);
	      const GeomDetUnit* theMonoDet = gluedDet->monoDet();
	      const GeomDetUnit* theStereoDet = gluedDet->stereoDet();
	      int monodetid = theMonoDet->geographicalId().rawId();
	      int stereodetid = theStereoDet->geographicalId().rawId();
	      //	      std::cout<< "   monodetid = " << monodetid << std::endl;
	      if( monodetid == (int) (*isim).detUnitId() && 
		  (int) (*iterrechit).simtrackId()==(int) (*isim).trackId()){
		//matching the rphi one
		//		std::cout<<"    ***  found matched matched hit ! ***"<< std::endl;
		matchedSimHits++;
		numpartners++;
		simHit = &(*isim);
		
		/*
		std::cout << "\tRecHit pos = " << rechit.localPosition() << "\tin Det " << detid.rawId() << "\tsimtkID = " << (*iterrechit).simtrackId() << std::endl;
		std::cout << "\tmatched to Simhit = " << (*isim).localPosition() << "\tsimtkId = " <<(*isim).trackId() << std::endl; 
		*/
	      }
	      if( stereodetid == (int) (*isim).detUnitId() && 
		  (int) (*iterrechit).simtrackId() == (int) (*isim).trackId()){
		//matching the rphi one
		numpartners++;
		//std::cout<<"    ***  found matched matched hit ! ***"<< std::endl;
		// 	matchedSimHits++;
		simHitStereoPatch = &(*isim);
		
		/*
		std::cout << "\tRecHit pos = " << rechit.localPosition() << "\tin Det " << detid.rawId() << "\tsimtkID = " << (*iterrechit).simtrackId() << std::endl;
		std::cout << "\tmatched to Simhit = " << (*isim).localPosition() << "\tsimtkId = " <<(*isim).trackId() << std::endl; 
		*/	  
	      }
	      
	      
	    }
	  }
	}
      }
      
      if(matchedSimHits==0 && simHitStereoPatch){
	matchedSimHits++;
	simHit=simHitStereoPatch;
      }

      if(matchedSimHits!=1 ){ // for debugging
	std::cout<<"ERROR: matchedSimHits!=1 " << std::endl;
	std::cout<<"ERROR: matchedSimHits =  " << matchedSimHits<<  std::endl;
	exit(1);
      }


      //make  plots for Hits
      makeHitsPlots("all_", &rechit, simHit, numpartners, tTopo);

    }
    //end all hits validation plots
    //----------------------------------------

    
    // number of pixel hits
    hMap["all_NumSimPixHits"]  ->Fill(iPixSimHits  );
    hMap["all_NumRecPixHits"]  ->Fill(iPixRecHits  );
    // number of strip hits
    hMap["all_NumSimStripHits"]->Fill(iStripSimHits);
    hMap["all_NumRecStripHits"]->Fill(iStripRecHits);
    // total number of hits
    hMap["all_NumRecHits"]->Fill(iTotalRecHits);
    hMap["all_NumSimHits"]->Fill(iTotalSimHits);
    // number of hits in track candidates
    hMap["cnd_NumRecHits"]->Fill(iCandRecHits);

//     //--------------------------------------
//     //start validation plots for tracks and hits in tracks
//     //---------------------------------------
     std::cout << "Reconstructed "<< tC.size() << " tracks" << std::endl ;
     std::cout << "Simulated "<< theSimTracks->size() << " tracks" << std::endl ;

     hMap["NumRecTracks"]->Fill(tC.size());
     hMap["NumSimTracks"]->Fill(theSimTracks->size());

    
     int i=1;
    for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){

      std::cout << "Track number "<< i << std::endl ;
      std::cout << "\tmomentum: " << track->momentum()<< "\tPT: " << track->pt()<< std::endl;
      std::cout << "\tvertex: " << track->vertex() << "\timpact parameter: " << track->d0()<< std::endl;
      std::cout << "\tcharge: " << track->charge() << "\tnormalizedChi2: " << track->normalizedChi2()<< std::endl;

      std::cout <<"\t\tNumber of RecHits "<<track->recHitsSize() << std::endl;
      SimTrackIds.clear();
      int ri=0;

       for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	ri++;
	if ((*it)->isValid()){

	  if(const SiTrackerGSRecHit2D * rechit = dynamic_cast<const SiTrackerGSRecHit2D *> (it->get()))	  
	    {
	      std::cout<<"---------------------------------------------------------------"<< std::endl;
	      int currentId = rechit->simtrackId();		      
	      std::cout << "\t\t\tRecHit # " << ri << "\t SimTrackId = " << currentId << std::endl;
	      SimTrackIds.push_back(currentId);
	      std::cout<<"\t\t\t SimHit ID belonging to this RecHit = "<< rechit->simhitId() << std::endl;
	      DetId detid = rechit->geographicalId();
	      unsigned int subdet = detid.subdetId();

	      // search the associated original PSimHit
	      unsigned int matchedSimHits = 0;
	      const PSimHit* simHit = NULL;
	      const PSimHit* simHitStereoPatch = NULL;
	      int numpartners=0;

	      for (MixCollection<PSimHit>::iterator isim=(*allTrackerHits).begin(); isim!= (*allTrackerHits).end(); isim++) {
		if( (int) detid.rawId() == (int) (*isim).detUnitId() && 
		    (int) rechit->simtrackId() == (int) (*isim).trackId()){
		  simHit = &(*isim);
		  matchedSimHits++;
		  /*
		  std::cout << "\tRecHit pos = " << rechit->localPosition() << "\tin Det " << detid.rawId() << "\tsimtkID = " << rechit->simtrackId() << std::endl;
		  std::cout << "\tmatched to Simhit = " << (*isim).localPosition() << "\tsimtkId = " <<(*isim).trackId() << std::endl; 	    		  
		  */
		}
		else{
		  if(subdet > 2){
		    //	    StripSubdetector specDetId=StripSubdetector(detid);
		    //	    std::cout<<"   glued = " << specDetId.glued() << std::endl;
		    const GluedGeomDet* gluedDet = dynamic_cast<const GluedGeomDet*> (trackerG->idToDet(detid));
		    //	    std::cout<<"   gluedgeomdet = " << gluedDet << std::endl;
		    if(gluedDet){
		      //	      const GluedGeomDet* gluedDet = (const GluedGeomDet*)trackerG->idToDet(detid);
		      const GeomDetUnit* theMonoDet = gluedDet->monoDet();
		      const GeomDetUnit* theStereoDet = gluedDet->stereoDet();
		      int monodetid = theMonoDet->geographicalId().rawId();
		      int stereodetid = theStereoDet->geographicalId().rawId();
		      //	      std::cout<< "   monodetid = " << monodetid << std::endl;
		      if( monodetid == (int) (*isim).detUnitId() && 
			  (int) rechit->simtrackId() == (int) (*isim).trackId()){
			//matching the rphi one
			//std::cout<<"    ***  found matched matched hit ! ***"<< std::endl;
			matchedSimHits++;
			numpartners++;
			simHit = &(*isim);
		

		      }
		      if( stereodetid == (int) (*isim).detUnitId() && 
			  (int) (*rechit).simtrackId() == (int) (*isim).trackId()){
			//matching the rphi one
			numpartners++;
			//std::cout<<"    ***  found matched matched hit ! ***"<< std::endl;
			// 	matchedSimHits++;
			simHitStereoPatch = &(*isim);
			

		      }
		      
		      
		    }
		  }
		}
		
	      }// end loop on simhits


	      if(matchedSimHits==0 && simHitStereoPatch){
		matchedSimHits++;
		simHit=simHitStereoPatch;
	      }
	      
	      if(matchedSimHits!=1 ){ // for debugging
		std::cout<<"ERROR: matchedSimHits!=1 " << std::endl;
		std::cout<<"ERROR: matchedSimHits =  " << matchedSimHits<<  std::endl;
		exit(1);
	      }
	      
	      
	      //make plots for Hits
	      makeHitsPlots("", rechit, simHit, numpartners, tTopo);
	 
	    }
	}else{
	  cout <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId()<<endl;
	} 
       }// end loop on rechits
       
       int nmax = 0;
       int idmax = -1;
       for(size_t j=0; j<SimTrackIds.size(); j++){
	 int n =0;
	 n = std::count(SimTrackIds.begin(), SimTrackIds.end(), SimTrackIds[j]);
	 if(n>nmax){
	  nmax = n;
	  idmax = SimTrackIds[i];
	 }
       }
       //       float totsim = nmax;
       //       float tothits = track->recHitsSize();//include pixel as well..
       //       float fraction = totsim/tothits ;
       
       //       std::cout << "Track id # " << i << "\tmatched to Simtrack id= " << idmax  << "\t momentum = " << track->momentum() << std::endl;
       //       std::cout << "\tN(matches)= " << totsim <<  "\t# of rechits = " << track->recHitsSize() 
       // 		<< "\tfraction = " << fraction << std::endl;
       
       
       //      //now found the simtrack information
       for(SimTrackContainer::const_iterator simTrack = theSimTracks->begin(); simTrack != theSimTracks->end(); simTrack++)
	 { 
	   if( (int) simTrack->trackId() == idmax) {
	     std::cout << "\t\tSim track mom = " << simTrack->momentum() << " charge = " <<  simTrack->charge() << std::endl;
	     
	     
	     
	     hMap["trk_Rec_chi2"]->Fill(track->chi2());
	     hMap["trk_Rec_Normchi2"]->Fill(track->normalizedChi2());
	     hMap["trk_Rec_ndof"]->Fill(track->ndof());
	     
	     hMap["trk_Rec_phi"]     ->Fill(  track->phi() );
	     hMap["trk_Sim_phi"] 	  ->Fill( simTrack->momentum().phi()); 
	     hMap["trk_Res_phi"] 	  ->Fill(  track->phi() -  simTrack->momentum().phi() );
	     hMap["trk_Pull_phi"]	  ->Fill( ( track->phi() -  simTrack->momentum().phi()) / track->phiError()   );
	     
	     hMap["trk_Rec_eta"] 	  ->Fill(    track->eta() );                                                      
	     hMap["trk_Sim_eta"] 	  ->Fill(   simTrack->momentum().eta());                                          
	     hMap["trk_Res_eta"] 	  ->Fill(    track->eta() -  simTrack->momentum().eta() );                        
	     hMap["trk_Pull_eta"]	  ->Fill(   ( track->eta() -  simTrack->momentum().eta()) / track->etaError()   );
	     
	     hMap["trk_Rec_pt"] 	  ->Fill( track->pt()  );
	     hMap["trk_Sim_pt"] 	  ->Fill( simTrack->momentum().Pt()  );
	     hMap["trk_Res_pt"] 	  ->Fill( track->pt() -  simTrack->momentum().Pt() );
	     
	     double simQoverp =  simTrack->charge() / simTrack->momentum().P();
	     
	     hMap["trk_Pull_qoverp"] ->Fill( (track->qoverp() -  simQoverp) / track->qoverpError()  );
	     std::cout<<" qoverp = " << track->qoverp() << std::endl;
	     std::cout<<" simqoverp = "<< simQoverp << std::endl;
	     std::cout<<" qoverpPull = "<<  (track->qoverp() -  simQoverp) / track->qoverpError() << std::endl;
	     
	     hMap["trk_Rec_d0"] 	  ->Fill(   track->d0() );
	     hMap["trk_Err_d0"]	  ->Fill(  track->d0Error()  );
	     //	    hMap["trk_Pull_d0"]	  ->Fill(   );
	     hMap["trk_Rec_dz"] 	  ->Fill(   track->dz() );
	     hMap["trk_Err_dz"]	  ->Fill(  track->dzError()  );
	     
	     hMap["trk_Rec_charge"] ->Fill( track->charge() );
	   }
	   
	   
	 }
    } // end loop on rectracks
    

    /*

// NEEDS TO BE FIXED PROPERLY 
    
    for(TrackCandidateCollection::const_iterator it = theTrackCandColl->begin(); it!= theTrackCandColl->end(); it++){
      PTrajectoryStateOnDet state = it->trajectoryStateOnDet();
      //convert PTrajectoryStateOnDet to TrajectoryStateOnSurface
      
      DetId  detId(state.detId());
      TrajectoryStateOnSurface theTSOS = transformer.transientState( state,
								     &(theG->idToDet(detId)->surface()), 
								     theMagField.product());
      
      hMap["trk_Cnd_eta"]->Fill(   theTSOS.globalMomentum().eta() );
    }

    */
    
  }


//------------------------------------------------------
void FastTrackAnalyzer::makeHitsPlots(TString prefix, const SiTrackerGSRecHit2D * rechit, const PSimHit * simHit, 
				      int numpartners, const TrackerTopology *tTopo){
  //  std::cout<< "making plots" << std::endl;

  DetId adetid = rechit->geographicalId();
  //  DetId simdetid= DetId(simHit->detUnitId());
  //  DetId recdetid = rechit->geographicalId();

  unsigned int subdet   = DetId(simHit->detUnitId()).subdetId();
  unsigned int detid    = DetId(simHit->detUnitId()).rawId();

  //  const GeomDetUnit *  det = trackerG->idToDetUnit(adetid);
  const GeomDet* det = trackerG->idToDet( adetid ); 
  // const GeomDetUnit *  simdet = trackerG->idToDetUnit(DetId(simHit->detUnitId()));

  const GluedGeomDet* gluedDet = dynamic_cast<const GluedGeomDet*> (trackerG->idToDet(adetid));

  GlobalPoint posGlobRec =  det->surface().toGlobal(  rechit->localPosition() );
  GlobalPoint posGlobSim =  det->surface().toGlobal( simHit->localPosition()  );
  

  float xGlobRec = posGlobRec.x();
  float yGlobRec = posGlobRec.y();
  float zGlobRec = posGlobRec.z();
  // float rGlobRec = posGlobRec.perp();

  float xGlobSim = posGlobSim.x();
  float yGlobSim = posGlobSim.y();
  float zGlobSim = posGlobSim.z();
  // float rGlobSim = posGlobSim.perp();

  float xRec = rechit->localPosition().x();
  float yRec = rechit->localPosition().y();
  // float zRec = rechit->localPosition().z();
  
  float xSim = simHit->localPosition().x();
  float ySim = simHit->localPosition().y();
  //  float zSim = simHit->localPosition().z();
 
  if(gluedDet){    
    const StripGeomDetUnit* partnerstripdet = (StripGeomDetUnit*) gluedDet->monoDet();
    //const StripGeomDetUnit* stereostripdet = (StripGeomDetUnit*) gluedDet->stereoDet();
    std::pair<LocalPoint,LocalVector> hitPair1, hitPair2;
    //check both track directions
    hitPair1= projectHit(*simHit,partnerstripdet,gluedDet->surface(), 1);
    hitPair2= projectHit(*simHit,partnerstripdet,gluedDet->surface(), -1);
    /*
    std::cout<<"   before after sim project: " << std::endl;
    std::cout<<"    xSimBefore = " << xSim << std::endl;
    std::cout<<"    ySimBefore = " << ySim << std::endl;
    */
    float xSim1 =  hitPair1.first.x();
    float ySim1 =  hitPair1.first.y();
    //float zSim1 =  hitPair1.first.z();

    float xSim2 =  hitPair2.first.x();
    float ySim2 =  hitPair2.first.y();
    //float zSim2 =  hitPair2.first.z();

    if( ((xSim1-xRec)*(xSim1-xRec)+(ySim1-yRec)*(ySim1-yRec)) <  ((xSim2-xRec)*(xSim2-xRec)+(ySim2-yRec)*(ySim2-yRec))){
      xSim =  hitPair1.first.x();
      ySim =  hitPair1.first.y();
      //      zSim =  hitPair1.first.z();
    }
    else{
      xSim =  hitPair2.first.x();
      ySim =  hitPair2.first.y();
      //      zSim =  hitPair2.first.z();
    }
    /*
    std::cout<<"    xSim1After = " << xSim1 << std::endl;
    std::cout<<"    ySim1After = " << ySim1 << std::endl;
    std::cout<<"    xSim2After = " << xSim2 << std::endl;
    std::cout<<"    ySim2After = " << ySim2 << std::endl;
    */
  }

  float delta_x = xRec - xSim;
  float delta_y = yRec - ySim;
  // float delta_z = zRec - zSim;
  
  float err_xx = sqrt(rechit->localPositionError().xx());
  float err_xy = sqrt(rechit->localPositionError().xy());
  float err_yy = sqrt(rechit->localPositionError().yy());


/*
  std::cout<<"plotting for prefix "<< prefix << std::endl;
  std::cout<<"\t\t\t detid = "<< detid << "  subdet = "<< subdet<<" which means: "<< std::endl;
*/ 

 switch (subdet) {
    // Pixel Barrel
  case 1: {
		
		unsigned int theLayer = tTopo->pxbLayer(detid);
		//	std::cout << "\t\t\tPixel Barrel Layer " << theLayer << std::endl;
		TString layer = ""; layer+=theLayer;
		hMap[prefix+"PXB_RecPos_x_"+layer]->Fill(xRec);
		hMap[prefix+"PXB_RecPos_y_"+layer]->Fill(yRec);
		hMap[prefix+"PXB_SimPos_x_"+layer]->Fill(xSim);
		hMap[prefix+"PXB_SimPos_y_"+layer]->Fill(ySim);
		hMap[prefix+"PXB_Res_x_"+layer]->Fill(delta_x);
		hMap[prefix+"PXB_Res_y_"+layer]->Fill(delta_y);
		hMap[prefix+"PXB_Err_xx"+layer]->Fill(err_xx);
		hMap[prefix+"PXB_Err_xy"+layer]->Fill(err_xy);
		hMap[prefix+"PXB_Err_yy"+layer]->Fill(err_yy);
		break;
	      }
		//Pixel Forward
	      case 2:    {
		
		unsigned int theDisk = tTopo->pxfDisk(detid);
		//std::cout << "\t\t\tPixel Forward Disk " << theDisk << std::endl;
		TString layer = ""; layer+=theDisk;
		hMap[prefix+"PXF_RecPos_x_"+layer]->Fill(xRec);
		hMap[prefix+"PXF_RecPos_y_"+layer]->Fill(yRec);
		hMap[prefix+"PXF_SimPos_x_"+layer]->Fill(xSim);
		hMap[prefix+"PXF_SimPos_y_"+layer]->Fill(ySim);
		hMap[prefix+"PXF_Res_x_"+layer]->Fill(delta_x);
		hMap[prefix+"PXF_Res_y_"+layer]->Fill(delta_y);
		hMap[prefix+"PXF_Err_xx"+layer]->Fill(err_xx);
		hMap[prefix+"PXF_Err_xy"+layer]->Fill(err_xy);
		hMap[prefix+"PXF_Err_yy"+layer]->Fill(err_yy);
		break;
	      }
		// TIB
	      case 3:
		{
		  
		  unsigned int theLayer  = tTopo->tibLayer(detid);
		  /*
		  std::cout << "\t\t\tTIB Layer " << theLayer << std::endl;
		  std::cout<<"\t\t\t recLocX = "<< xRec << "   simLocX = "<< xSim << std::endl;
		  std::cout<<"\t\t\t recLocY = "<< yRec << "   simLocY = "<< ySim << std::endl;
		  std::cout<<"\t\t\t recLocZ = "<< zRec << "   simLocZ = "<< zSim << std::endl;
		  std::cout<<"\t\t\t errX = "<< err_xx <<"   errY = "<< err_yy << std::endl;
		  */
		  TString layer=""; layer+=theLayer;
		  hMap[prefix+"TIB_Res_x_"+layer] ->Fill(delta_x);
		  hMap[prefix+"TIB_SimPos_x_"+layer] ->Fill(xSim);
 		  hMap[prefix+"TIB_RecPos_x_"+layer] ->Fill(xRec);
		  hMap[prefix+"TIB_Err_x_"+layer] ->Fill(err_xx);
		  if(theLayer ==1 || theLayer ==2){
		    hMap[prefix+"TIB_Res_y_"+layer] ->Fill(delta_y);
		    hMap[prefix+"TIB_RecPos_y_"+layer] ->Fill(yRec);
		    hMap[prefix+"TIB_SimPos_y_"+layer] ->Fill(ySim);
		    hMap[prefix+"TIB_Err_y_"+layer] ->Fill(err_yy);
		  }
		  break;
		}
		// TID
	      case 4:
		{
		  
		  unsigned int theRing  = tTopo->tidRing(detid);
		  //std::cout << "\t\t\tTID Ring " << theRing << std::endl;
		  TString ring=""; ring+=theRing;

		  hMap[prefix+"TID_Res_x_"+ring] ->Fill(delta_x);
		  if(gluedDet) if( numpartners == 1) {
		    hMap[prefix+"TID_Res_x_proj_"+ring]->Fill(delta_x);
		  }
		  hMap[prefix+"TID_SimPos_x_"+ring] ->Fill(xSim);
		  hMap[prefix+"TID_RecPos_x_"+ring] ->Fill(xRec);
		  hMap[prefix+"TID_Err_x_"+ring] ->Fill(err_xx);

		  if(theRing==1 || theRing==2){
		    hMap[prefix+"TID_Res_y_"+ring] ->Fill(delta_y);
		    hMap[prefix+"TID_RecPos_y_"+ring] ->Fill(yRec);
		    hMap[prefix+"TID_SimPos_y_"+ring] ->Fill(ySim);
		    hMap[prefix+"TID_Err_y_"+ring] ->Fill(err_yy);
		  }
		  break;
		}
		    // TOB
	      case 5:
		{
		  
		  unsigned int theLayer  = tTopo->tobLayer(detid);
		  //std::cout << "\t\t\tTOB Layer " << theLayer << std::endl;
		  TString layer=""; layer+=theLayer;
		  hMap[prefix+"TOB_Res_x_"+layer] ->Fill(delta_x);
		  hMap[prefix+"TOB_SimPos_x_"+layer] ->Fill(xSim);
		  hMap[prefix+"TOB_RecPos_x_"+layer] ->Fill(xRec);
		  hMap[prefix+"TOB_Err_x_"+layer] ->Fill(err_xx);
		  if(theLayer ==1 || theLayer ==2){
		    hMap[prefix+"TOB_Res_y_"+layer] ->Fill(delta_y);
		    hMap[prefix+"TOB_RecPos_y_"+layer] ->Fill(yRec);
		    hMap[prefix+"TOB_SimPos_y_"+layer] ->Fill(ySim);
		    hMap[prefix+"TOB_Err_y_"+layer] ->Fill(err_yy);
		  }
		  break;
		}
		// TEC
	      case 6:
		{

		  //	   StripSubdetector specDetId=StripSubdetector(detid);
		  
		  unsigned int theRing  = tTopo->tecRing(detid);
		  //unsigned int theWheel = tTopo->tecWheel(detid);
		  if(!gluedDet && theRing==1){
		    std::cout<<"     AS debugging2 !gluedDet && theRing==1"<< std::endl;
		    // exit(1);
		  }
		  //std::cout << "\t\t\tTEC Ring " << theRing << ",    wheel = " << theWheel <<  std::endl;
		  TString ring=""; ring+=theRing;
		  

		    hMap[prefix+"TEC_Res_x_"+ring] ->Fill(delta_x);
		    hMap[prefix+"TEC_SimPos_x_"+ring] ->Fill(xSim);
		    hMap[prefix+"TEC_RecPos_x_"+ring] ->Fill(xRec);
		    hMap[prefix+"TEC_Err_x_"+ring] ->Fill(err_xx);

		  if(theRing ==1 || theRing ==2 || theRing==5){
		    hMap[prefix+"TEC_Res_y_"+ring] ->Fill(delta_y);
		    hMap[prefix+"TEC_RecPos_y_"+ring] ->Fill(yRec);
		    hMap[prefix+"TEC_SimPos_y_"+ring] ->Fill(ySim);
		    hMap[prefix+"TEC_Err_y_"+ring] ->Fill(err_yy);

		    if(gluedDet) if( numpartners==1) {
		      hMap[prefix+"TEC_Res_x_proj_"+ring]->Fill(delta_x);
		    }

		  }

		  break;
		}
		
	      }
	  
	      hMap["all_GlobSimPos_x"] ->Fill(xGlobSim);
	      hMap["all_GlobSimPos_y"] ->Fill(yGlobSim);
	      hMap["all_GlobSimPos_z"] ->Fill(zGlobSim);

	      hMap["all_GlobRecPos_x"] ->Fill( xGlobRec );
	      hMap["all_GlobRecPos_y"] ->Fill( yGlobRec);
	      hMap["all_GlobRecPos_z"] ->Fill( zGlobRec);



}





void FastTrackAnalyzer::endJob(){

  TFile* outfile = new TFile(outfilename, "RECREATE");
  outfile->cd();
  for(std::map<TString, TH1F*>::iterator it = hMap.begin(); it!=hMap.end(); it++){
    it->second->Write();
  }
  outfile->Close();
}


std::pair<LocalPoint,LocalVector> FastTrackAnalyzer::projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet,
								   const BoundPlane& plane, int thesign) 
{
  //  const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(hit.det());
  //if (stripDet == 0) throw MeasurementDetException("HitMatcher hit is not on StripGeomDetUnit");
  
  const StripTopology& topol = stripDet->specificTopology();
  GlobalPoint globalpos= stripDet->surface().toGlobal(hit.localPosition());
  LocalPoint localHit = plane.toLocal(globalpos);
  //track direction
  //  LocalVector locdir=hit.localDirection();

  LocalPoint lcenterofstrip=hit.localPosition();
  GlobalPoint gcenterofstrip=(stripDet->surface()).toGlobal(lcenterofstrip);
  GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
 LocalVector dir= plane.toLocal(gtrackdirection);

  //rotate track in new frame
  
  //  GlobalVector globaldir= stripDet->surface().toGlobal(locdir);
  //  LocalVector dir=plane.toLocal(globaldir);
  float scale = thesign * localHit.z() / dir.z();
    LocalPoint projectedPos = localHit + scale*dir;
  
  //  std::cout << "projectedPos " << projectedPos << std::endl;
  
  float selfAngle = topol.stripAngle( topol.strip( hit.localPosition()));
  
  LocalVector stripDir( sin(selfAngle), cos(selfAngle), 0); // vector along strip in hit frame
  
  LocalVector localStripDir( plane.toLocal(stripDet->surface().toGlobal( stripDir)));
  
  return std::pair<LocalPoint,LocalVector>( projectedPos, localStripDir);
}


//define this as a plug-in

DEFINE_FWK_MODULE(FastTrackAnalyzer);

