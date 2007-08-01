//Alexander.Schmidt@cern.ch
//March 2007

// This code was created during the debugging of the fast RecHits and fast 
// tracks.  It produces some validation and debugging plots of the RecHits and tracks.

#include "FastSimulation/Tracking/test/FastTrackAnalyzer.h"
#include "Math/GenVector/BitReproducible.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h" 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

// Numbering scheme
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "TH1F.h"
#include "TFile.h"
#include "TString.h"
#include <memory>
#include <iostream>
#include <string>

using namespace edm;
using namespace std;
    
//---------------------------------------------------------
FastTrackAnalyzer::FastTrackAnalyzer(edm::ParameterSet const& conf) : 
  conf_(conf) {
  
  iEventCounter=0;
  
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
  TIB_Pos_AxisLim = conf.getParameter<double>("TIB_Pos_AxisLim" );
  TID_Res_AxisLim = conf.getParameter<double>("TID_Res_AxisLim" );
  TID_Pos_AxisLim = conf.getParameter<double>("TID_Pos_AxisLim" );
  TOB_Res_AxisLim = conf.getParameter<double>("TOB_Res_AxisLim" );
  TOB_Pos_AxisLim = conf.getParameter<double>("TOB_Pos_AxisLim" );
  TEC_Res_AxisLim = conf.getParameter<double>("TEC_Res_AxisLim" );
  TEC_Pos_AxisLim = conf.getParameter<double>("TEC_Pos_AxisLim" );
  
  TIB_Err_AxisLim  = conf.getParameter<double>("TIB_Err_AxisLim" );
  TID_Err_AxisLim  = conf.getParameter<double>("TID_Err_AxisLim" );
  TOB_Err_AxisLim  = conf.getParameter<double>("TOB_Err_AxisLim" );
  TEC_Err_AxisLim  = conf.getParameter<double>("TEC_Err_AxisLim" );

  NumTracks_AxisLim = conf.getParameter<int>("NumTracks_AxisLim" );
}
//---------------------------------------------------------
FastTrackAnalyzer::~FastTrackAnalyzer() {}
//---------------------------------------------------------
void FastTrackAnalyzer::beginJob( const edm::EventSetup& es){
  
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

  // strip maximum 9 layers (TEC)
  for(int i=1; i<=9; i++){
    TString index = ""; index+=i;
    if(i<5){
      hMap["all_TIB_Res_x_"+index] = new TH1F("all_TIB_Res_x_"+index, "all_TIB_Res_x_"+index, 100, -TIB_Res_AxisLim, TIB_Res_AxisLim);
      hMap["all_TIB_SimPos_x_"+index] = new TH1F("all_TIB_SimPos_x_"+index, "all_TIB_SimPos_x_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
      hMap["all_TIB_RecPos_x_"+index] = new TH1F("all_TIB_RecPos_x_"+index, "all_TIB_RecPos_x_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
      hMap["all_TIB_Err_x_"+index] = new TH1F("all_TIB_Err_x_"+index, "all_TIB_Err_x_"+index, 100, -TIB_Err_AxisLim, TIB_Err_AxisLim);
      hMap["TIB_Res_x_"+index] = new TH1F("TIB_Res_x_"+index, "TIB_Res_x_"+index, 100, -TIB_Res_AxisLim, TIB_Res_AxisLim);
      hMap["TIB_SimPos_x_"+index] = new TH1F("TIB_SimPos_x_"+index, "TIB_SimPos_x_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
      hMap["TIB_RecPos_x_"+index] = new TH1F("TIB_RecPos_x_"+index, "TIB_RecPos_x_"+index, 100, -TIB_Pos_AxisLim, TIB_Pos_AxisLim);
      hMap["TIB_Err_x_"+index] = new TH1F("TIB_Err_x_"+index, "TIB_Err_x_"+index, 100, -TIB_Err_AxisLim, TIB_Err_AxisLim);
    }
    if(i<7){
      hMap["all_TOB_Res_x_"+index] = new TH1F("all_TOB_Res_x_"+index, "all_TOB_Res_x_"+index, 100, -TOB_Res_AxisLim, TOB_Res_AxisLim);
      hMap["all_TOB_SimPos_x_"+index] = new TH1F("all_TOB_SimPos_x_"+index, "all_TOB_SimPos_x_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
      hMap["all_TOB_RecPos_x_"+index] = new TH1F("all_TOB_RecPos_x_"+index, "all_TOB_RecPos_x_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
      hMap["all_TOB_Err_x_"+index] = new TH1F("all_TOB_Err_x_"+index, "all_TOB_Err_x_"+index, 100, -TOB_Err_AxisLim, TOB_Err_AxisLim);
      hMap["TOB_Res_x_"+index] = new TH1F("TOB_Res_x_"+index, "TOB_Res_x_"+index, 100, -TOB_Res_AxisLim, TOB_Res_AxisLim);
      hMap["TOB_SimPos_x_"+index] = new TH1F("TOB_SimPos_x_"+index, "TOB_SimPos_x_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
      hMap["TOB_RecPos_x_"+index] = new TH1F("TOB_RecPos_x_"+index, "TOB_RecPos_x_"+index, 100, -TOB_Pos_AxisLim, TOB_Pos_AxisLim);
      hMap["TOB_Err_x_"+index] = new TH1F("TOB_Err_x_"+index, "TOB_Err_x_"+index, 100, -TOB_Err_AxisLim, TOB_Err_AxisLim);
    }

    hMap["all_TEC_Res_x_"+index] = new TH1F("all_TEC_Res_x_"+index, "all_TEC_Res_x_"+index, 100, -TEC_Res_AxisLim, TEC_Res_AxisLim);
    hMap["all_TEC_SimPos_x_"+index] = new TH1F("all_TEC_SimPos_x_"+index, "all_TEC_SimPos_x_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
    hMap["all_TEC_RecPos_x_"+index] = new TH1F("all_TEC_RecPos_x_"+index, "all_TEC_RecPos_x_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
    hMap["all_TEC_Err_x_"+index] = new TH1F("all_TEC_Err_x_"+index, "all_TEC_Err_x_"+index, 100, -TEC_Err_AxisLim, TEC_Err_AxisLim);
    hMap["TEC_Res_x_"+index] = new TH1F("TEC_Res_x_"+index, "TEC_Res_x_"+index, 100, -TEC_Res_AxisLim, TEC_Res_AxisLim);
    hMap["TEC_SimPos_x_"+index] = new TH1F("TEC_SimPos_x_"+index, "TEC_SimPos_x_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
    hMap["TEC_RecPos_x_"+index] = new TH1F("TEC_RecPos_x_"+index, "TEC_RecPos_x_"+index, 100, -TEC_Pos_AxisLim, TEC_Pos_AxisLim);
    hMap["TEC_Err_x_"+index] = new TH1F("TEC_Err_x_"+index, "TEC_Err_x_"+index, 100, -TEC_Err_AxisLim, TEC_Err_AxisLim);

    if(i<4){
      hMap["all_TID_Res_x_"+index] = new TH1F("all_TID_Res_x_"+index, "all_TID_Res_x_"+index, 100, -TID_Res_AxisLim, TID_Res_AxisLim);
      hMap["all_TID_SimPos_x_"+index] = new TH1F("all_TID_SimPos_x_"+index, "all_TID_SimPos_x_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
      hMap["all_TID_RecPos_x_"+index] = new TH1F("all_TID_RecPos_x_"+index, "all_TID_RecPos_x_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
      hMap["all_TID_Err_x_"+index] = new TH1F("all_TID_Err_x_"+index, "all_TID_Err_x_"+index, 100, -TID_Err_AxisLim, TID_Err_AxisLim);
      hMap["TID_Res_x_"+index] = new TH1F("TID_Res_x_"+index, "TID_Res_x_"+index, 100, -TID_Res_AxisLim, TID_Res_AxisLim);
      hMap["TID_SimPos_x_"+index] = new TH1F("TID_SimPos_x_"+index, "TID_SimPos_x_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
      hMap["TID_RecPos_x_"+index] = new TH1F("TID_RecPos_x_"+index, "TID_RecPos_x_"+index, 100, -TID_Pos_AxisLim, TID_Pos_AxisLim);
      hMap["TID_Err_x_"+index] = new TH1F("TID_Err_x_"+index, "TID_Err_x_"+index, 100, -TID_Err_AxisLim, TID_Err_AxisLim);
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

    std::cout << "\nEvent ID = "<< event.id() << std::endl ;
    
    // get rec track collection
    edm::Handle<reco::TrackCollection> trackCollection;
    event.getByType(trackCollection);
    const reco::TrackCollection tC = *(trackCollection.product());
    
    //get simtrack info
    edm::Handle<SimTrackContainer> theSimTracks;
    event.getByLabel("famosSimHits",theSimTracks);

    edm::Handle<SimVertexContainer> theSimVtx;
    event.getByLabel("famosSimHits",theSimVtx);

    // print size of vertex collection
    std::cout<<" AS: vertex.size() = "<< theSimVtx->size() << std::endl;

    // get track candidates
    edm::Handle<TrackCandidateCollection> theTrackCandidates;
    event.getByType(theTrackCandidates);
    const TrackCandidateCollection* theTrackCandColl = theTrackCandidates.product();

    std::vector<unsigned int> SimTrackIds;


    // Get PSimHit's of the Event
    edm::Handle<CrossingFrame> cf;
    event.getByType(cf);
    MixCollection<PSimHit> allTrackerHits(cf.product(),trackerContainers);

    //Get RecHits from the event
    edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
    event.getByType(theGSRecHits);
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
    
    // count number of hits in track candidate
    // stop with error if no candidate
    if(theTrackCandColl->size() != 1){
      std::cout<<"  AS: ERROR debugging: theTrackCandColl->size() != 1"<< std::endl;
      exit(1);
    }
    for(TrackCandidateCollection::const_iterator it = theTrackCandColl->begin(); it!= theTrackCandColl->end(); it++){
      TrackCandidate::RecHitContainer::const_iterator theItBegin =   it->recHits().first;
      TrackCandidate::RecHitContainer::const_iterator theItEnd =   it->recHits().second;
      for( TrackCandidate::RecHitContainer::const_iterator theIt = theItBegin; theIt!= theItEnd; theIt++){
	iCandRecHits++;
      }
    }

//     // print position of simhits for debugging
//     std::cout<<" AS: global position of all simhits (x, y, z, r)"<< std::endl;
//     //loop on all simhits to count them
//     for (MixCollection<PSimHit>::iterator isim=allTrackerHits.begin(); isim!= allTrackerHits.end(); isim++) {
//       unsigned int subdet   = DetId(isim->detUnitId()).subdetId();
//       iTotalSimHits++;
//       if(subdet==1 || subdet==2)iPixSimHits++;
//       else if(subdet==3|| subdet==4 || subdet==5 || subdet == 6) iStripSimHits++;
//       else {
// 	std::cout<<" AS: ERROR simhit subdet inconsistent"<< std::endl;
// 	exit(1);
//       }

//       const GeomDetUnit *  det = tracker.idToDetUnit( DetId(isim->detUnitId())  );
//       GlobalPoint posGlobSim =     det->surface().toGlobal( isim->localPosition()  );
      
//       std::cout<< posGlobSim.x() << "  ,  " << posGlobSim.y() << "  ,  "<< posGlobSim.z() << "  ,  " << posGlobSim.perp() << std::endl;
//     }

    
    //------------------------
    //loop on all rechits, match them to simhits and make plots
    for(edm::OwnVector<SiTrackerGSRecHit2D>::const_iterator iterrechit = theRecHitIteratorBegin;
	iterrechit != theRecHitIteratorEnd; ++iterrechit) { // loop on GSRecHits
      
      DetId detid = (*iterrechit).geographicalId();

      // count rechits
      unsigned int subdet = detid.subdetId();
      iTotalRecHits++;
      if(subdet==1 || subdet==2) iPixRecHits++;
      else if(subdet==3|| subdet==4 || subdet==5 || subdet == 6) iStripRecHits++;
      else { // for debugging
	std::cout<<" AS: ERROR simhit subdet inconsistent"<< std::endl;
	exit(1);
      }

      SiTrackerGSRecHit2D const rechit=*iterrechit;

      LocalPoint position=rechit.localPosition();
      LocalError error=rechit.localPositionError();

      //match sim hit to rec hit
      unsigned int matchedSimHits = 0;
      PSimHit* simHit = NULL;
      for (MixCollection<PSimHit>::iterator isim=allTrackerHits.begin(); isim!= allTrackerHits.end(); isim++) {
	//compare detUnitIds && SimTrackIds to match rechit to simhit (for pileup will need to add EncodedEventId info as well).
	int simdetid = (*isim).detUnitId();
	if(detid.rawId() == (*isim).detUnitId() && (*iterrechit).simtrackId()==(*isim).trackId())
	  {
	    matchedSimHits++;
	    simHit = const_cast<PSimHit*>(&(*isim));
	    std::cout << "\tRecHit pos = " << position << "\tin Det " << detid.rawId() << "\tsimtkID = " << (*iterrechit).simtrackId() << std::endl;
	    std::cout << "\tmatched to Simhit = " << (*isim).localPosition() << "\tsimtkId = " <<(*isim).trackId() << std::endl; 	    
	  }
      }
      if(matchedSimHits!=1){ // for debugging
	std::cout<<"ERROR: matchedSimHits!=1 " << std::endl;
	exit(1);
      }

      // now match TrackingRecHits from track candidate to rec hit
      unsigned int matchedCandHits = 0;
      const TrackingRecHit * matchedCandHit;
      TrackCandidate::RecHitContainer::const_iterator theItBegin =    theTrackCandColl->begin()->recHits().first;
      TrackCandidate::RecHitContainer::const_iterator theItEnd =    theTrackCandColl->begin()->recHits().second;
      for( TrackCandidate::RecHitContainer::const_iterator theIt = theItBegin; theIt!= theItEnd; theIt++){
	const TrackingRecHit & candHit = *theIt;
	if(detid.rawId() == candHit.geographicalId().rawId()){
	  matchedCandHits++;
	  matchedCandHit = &candHit;
	}
      }
      if(matchedCandHits!=1){
	std::cout<<"  AS:ERROR: matchedCandHits!=1"<< std::endl;
	exit(1);
      }


      //make  plots for Hits
      makeHitsPlots("all_", &rechit, simHit, matchedCandHit);

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

    //--------------------------------------
    //start validation plots for tracks and hits in tracks
    //---------------------------------------
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
	      DetId adetid = rechit->geographicalId();

	      // search the associated original PSimHit
	      PSimHit* simHit = NULL;
	      unsigned int matchedSimHits = 0;
	      for (MixCollection<PSimHit>::iterator isim=allTrackerHits.begin(); isim!= allTrackerHits.end(); isim++) {
		if(adetid.rawId() == (*isim).detUnitId() && rechit->simtrackId()==(*isim).trackId()){
		  simHit = const_cast<PSimHit*>(&(*isim));
		  matchedSimHits++;
		  std::cout << "\tRecHit pos = " << rechit->localPosition() << "\tin Det " << adetid.rawId() << "\tsimtkID = " << rechit->simtrackId() << std::endl;
		  std::cout << "\tmatched to Simhit = " << (*isim).localPosition() << "\tsimtkId = " <<(*isim).trackId() << std::endl; 	    		  
		}
	      }
	      if(matchedSimHits!=1){
		std::cout<<"ERROR: matchedSimHits!=1 " << std::endl;
		exit(1);
	      }
	      
	      //make plots for Hits
	      makeHitsPlots("", rechit, simHit,0);
	 
	    }
	}else{
	  cout <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId()<<endl;
	} 
      }
      
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
      float totsim = nmax;
      float tothits = track->recHitsSize();//include pixel as well..
      float fraction = totsim/tothits ;
      
      std::cout << "Track id # " << i << "\tmatched to Simtrack id= " << idmax  << "\t momentum = " << track->momentum() << std::endl;
      std::cout << "\tN(matches)= " << totsim <<  "\t# of rechits = " << track->recHitsSize() 
		<< "\tfraction = " << fraction << std::endl;

      //now found the simtrack information
      for(SimTrackContainer::const_iterator simTrack = theSimTracks->begin(); simTrack != theSimTracks->end(); simTrack++)
	{ 
	  if(simTrack->trackId() == idmax) {
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
	    hMap["trk_Sim_pt"] 	  ->Fill( simTrack->momentum().perp()  );
	    hMap["trk_Res_pt"] 	  ->Fill( track->pt() -  simTrack->momentum().perp() );
	    
	    double simQoverp =  simTrack->charge() / simTrack->momentum().vect().mag();
	    
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
      i++;
      
    }// end loop on rec tracks

    ///--------------------
    /// compare first sim track to first track candidate
    const TrackCandidate & trackCand = *theTrackCandColl->begin();
    PTrajectoryStateOnDet ptrajState = trackCand.trajectoryStateOnDet();
    TrajectoryStateTransform transformer;
  
    DetId firstHitId(trackCand.recHits().first->geographicalId() );
    if(firstHitId.rawId() != ptrajState.detId()){
      std::cout<<"  AS:ERR: firstHitId.rawId() != ptrajState.detId() " << std::endl;
      exit(1);
    }
    cout<<" first hit id = "<< firstHitId.rawId() << std::endl;
    

    // trajectory state on first detector surface
    TrajectoryStateOnSurface  tsos = transformer.transientState(ptrajState, 
								&( tracker.idToDetUnit(	 firstHitId )->surface()),
								theMagField.product()) ;

    std::cout<<"\t  AS: simtrack compared to track candidate:"<< std::endl;
    std::cout<<" tsos.globalPosition().eta() = "<< tsos.globalPosition().eta() << std::endl;
    std::cout<<" tsos.globalPosition().phi() = "<< tsos.globalPosition().phi() << std::endl;
    std::cout<<" tsos.globalPosition().x() = "<< tsos.globalPosition().x() << std::endl;
    std::cout<<" tsos.globalPosition().y() = "<< tsos.globalPosition().y() << std::endl;
    std::cout<<" tsos.globalPosition().z() = "<< tsos.globalPosition().z() << std::endl<< std::endl;;
    std::cout<<" tsos.globalMomentum().eta() = "<< tsos.globalMomentum().eta() << std::endl;
    std::cout<<" tsos.globalMomentum().phi() = "<< tsos.globalMomentum().phi() << std::endl;
    std::cout<<" tsos.globalMomentum().x() = "<< tsos.globalMomentum().x() << std::endl;
    std::cout<<" tsos.globalMomentum().y() = "<< tsos.globalMomentum().y() << std::endl;
    std::cout<<" tsos.globalMomentum().z() = "<< tsos.globalMomentum().z() << std::endl<< std::endl;
    std::cout<<" theSimTracks->begin()->momentum().eta() = "<<    theSimTracks->begin()->momentum().eta() << std::endl;
    std::cout<<" theSimTracks->begin()->momentum().phi() = "<<    theSimTracks->begin()->momentum().phi() << std::endl;
    std::cout<<" theSimTracks->begin()->momentum().x() = "<<    theSimTracks->begin()->momentum().x() << std::endl;
    std::cout<<" theSimTracks->begin()->momentum().y() = "<<    theSimTracks->begin()->momentum().y() << std::endl;
    std::cout<<" theSimTracks->begin()->momentum().z() = "<<    theSimTracks->begin()->momentum().z() << std::endl;
  
    std::cout<<" vertex:"<< std::endl;
    std::cout<<" x = "<< (*theSimVtx)[0].position().x() << std::endl;
    std::cout<<" y = "<< (*theSimVtx)[0].position().y() << std::endl;
    std::cout<<" z = "<< (*theSimVtx)[0].position().z() << std::endl;
    //---------------------
    

  }


//------------------------------------------------------
void FastTrackAnalyzer::makeHitsPlots(TString prefix, const SiTrackerGSRecHit2D * rechit, PSimHit * simHit, const TrackingRecHit * candHit){
  DetId adetid = rechit->geographicalId();
  DetId simdetid= DetId(simHit->detUnitId());

  const GeomDetUnit *  det = trackerG->idToDetUnit(adetid);
  const GeomDetUnit *  simdet = trackerG->idToDetUnit(simdetid);

  GlobalPoint posGlobRec =  det->surface().toGlobal(  rechit->localPosition() );
  GlobalPoint posGlobSim =  det->surface().toGlobal( simHit->localPosition()  );

  float xGlobRec = posGlobRec.x();
  float yGlobRec = posGlobRec.y();
  float zGlobRec = posGlobRec.z();
  float rGlobRec = posGlobRec.perp();

  float xGlobSim = posGlobSim.x();
  float yGlobSim = posGlobSim.y();
  float zGlobSim = posGlobSim.z();
  float rGlobSim = posGlobSim.perp();

  float xRec = rechit->localPosition().x();
  float yRec = rechit->localPosition().y();
  float zRec = rechit->localPosition().z();
  
  float xSim = simHit->localPosition().x();
  float ySim = simHit->localPosition().y();
  float zSim = simHit->localPosition().z();
  
  float delta_x = xRec - xSim;
  float delta_y = yRec - ySim;
  float delta_z = zRec - zSim;
  
  float err_xx = sqrt(rechit->localPositionError().xx());
  float err_xy = sqrt(rechit->localPositionError().xy());
  float err_yy = sqrt(rechit->localPositionError().yy());

  DetId recdetid = rechit->geographicalId();
  unsigned int recsubdet = recdetid.subdetId();
  unsigned int subdet   = DetId(simHit->detUnitId()).subdetId();
  unsigned int detid    = DetId(simHit->detUnitId()).rawId();

  //check if rec and sim are the same
  if(subdet!=recsubdet){
    std::cout<<"subdet!=recsubdet"<<std::endl;
    exit(1);
  }
  std::cout<<"plotting for prefix "<< prefix << std::endl;
  std::cout<<"\t\t\t detid = "<< detid << "  subdet = "<< subdet<<" which means: "<< std::endl;
  switch (subdet) {
    // Pixel Barrel
  case 1: {
		PXBDetId module(detid);
		unsigned int theLayer = module.layer();
		std::cout << "\t\t\tPixel Barrel Layer " << theLayer << std::endl;
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
		PXFDetId module(detid);
		unsigned int theDisk = module.disk();
		std::cout << "\t\t\tPixel Forward Disk " << theDisk << std::endl;
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
		  TIBDetId module(detid);
		  unsigned int theLayer  = module.layer();
		  std::cout << "\t\t\tTIB Layer " << theLayer << std::endl;
		  TString layer=""; layer+=theLayer;
		  hMap[prefix+"TIB_Res_x_"+layer] ->Fill(delta_x);
		  hMap[prefix+"TIB_SimPos_x_"+layer] ->Fill(xSim);
		  hMap[prefix+"TIB_RecPos_x_"+layer] ->Fill(xRec);
		  hMap[prefix+"TIB_Err_x_"+layer] ->Fill(err_xx);

		  break;
		}
		// TID
	      case 4:
		{
		  TIDDetId module(detid);
		  unsigned int theRing  = module.ring();
		  std::cout << "\t\t\tTID Ring " << theRing << std::endl;
		  TString ring=""; ring+=theRing;
		  hMap[prefix+"TID_Res_x_"+ring] ->Fill(delta_x);
		  hMap[prefix+"TID_SimPos_x_"+ring] ->Fill(xSim);
		  hMap[prefix+"TID_RecPos_x_"+ring] ->Fill(xRec);
		  hMap[prefix+"TID_Err_x_"+ring] ->Fill(err_xx);
		  break;
		}
		    // TOB
	      case 5:
		{
		  TOBDetId module(detid);
		  unsigned int theLayer  = module.layer();
		  std::cout << "\t\t\tTOB Layer " << theLayer << std::endl;
		  TString layer=""; layer+=theLayer;
		  hMap[prefix+"TOB_Res_x_"+layer] ->Fill(delta_x);
		  hMap[prefix+"TOB_SimPos_x_"+layer] ->Fill(xSim);
		  hMap[prefix+"TOB_RecPos_x_"+layer] ->Fill(xRec);
		  hMap[prefix+"TOB_Err_x_"+layer] ->Fill(err_xx);
		  break;
		}
		// TEC
	      case 6:
		{
		  TECDetId module(detid);
		  unsigned int theRing  = module.ring();
		  unsigned int theWheel = module.wheel();
		  std::cout << "\t\t\tTEC Ring " << theRing << ",    wheel = " << theWheel <<  std::endl;
		  TString ring=""; ring+=theRing;
		  hMap[prefix+"TEC_Res_x_"+ring] ->Fill(delta_x);
		  hMap[prefix+"TEC_SimPos_x_"+ring] ->Fill(xSim);
		  hMap[prefix+"TEC_RecPos_x_"+ring] ->Fill(xRec);
		  hMap[prefix+"TEC_Err_x_"+ring] ->Fill(err_xx);
		  break;
		}
		
	      }
	  
	      float err_x = sqrt(rechit->localPositionError().xx());
	      float err_y = sqrt(rechit->localPositionError().yy());

	      // some debugging printout

 // 	      std::cout<<"\t\t\t recLocX = "<< xRec << "   simLocX = "<< xSim << std::endl;
//  	      std::cout<<"\t\t\t recLocY = "<< yRec << "   simLocY = "<< ySim << std::endl;
//  	      std::cout<<"\t\t\t recLocZ = "<< zRec << "   simLocZ = "<< zSim << std::endl;
//  	      std::cout<<"\t\t\t errX = "<< err_x <<"   errY = "<< err_y << std::endl;

// 	      std::cout<<"\t\t\t recGlobX = "<< xGlobRec << "   simGlobX = "<< xGlobSim << std::endl;
// 	      std::cout<<"\t\t\t recGlobY = "<< yGlobRec << "   simGlobY = "<< yGlobSim << std::endl;
// 	      std::cout<<"\t\t\t recGlobZ = "<< zGlobRec << "   simGlobZ = "<< zGlobSim << std::endl;
// 	      std::cout<<"\t\t\t recGlobR = "<< rGlobRec << "   simGlobR = "<< rGlobSim << std::endl;
// 	      std::cout<<"\t\t\t errX = "<< err_x <<"   errY = "<< err_y << std::endl;

// 	      if(candHit){
// 		DetId adetid = candHit->geographicalId();
// 		const GeomDetUnit *  det = trackerG->idToDetUnit(adetid);
		
// 		std::cout<<"\t\t\t candGlob_x = "<<  det->surface().toGlobal(  candHit->localPosition() ).x() << std::endl;
// 		std::cout<<"\t\t\t candGlob_y = "<<  det->surface().toGlobal(  candHit->localPosition() ).y() << std::endl;
// 		std::cout<<"\t\t\t candGlob_z = "<<  det->surface().toGlobal(  candHit->localPosition() ).z() << std::endl;
// 	      }


// 	      std::cout<<"\t\t\t surface of this hit: "<< std::endl;
// 	      std::cout<<"\t\t\t position = "<<      det->surface().position() << std::endl;
// 	      std::cout<<"\t\t\t position x = "<<      det->surface().position().x() << std::endl;
// 	      std::cout<<"\t\t\t position y = "<<      det->surface().position().y() << std::endl;
// 	      std::cout<<"\t\t\t position z = "<<      det->surface().position().z() << std::endl;
// 	      std::cout<<"\t\t\t position r = "<<      det->surface().position().perp() << std::endl;
// 	      std::cout<<"\t\t\t width = " << det->surface().bounds().width() << std::endl;
// 	      std::cout<<"\t\t\t length = " << det->surface().bounds().length() << std::endl;
// 	      std::cout<<"\t\t\t thickness = " << det->surface().bounds().thickness() << std::endl;
	    
	      
	      hMap["all_GlobSimPos_x"] ->Fill(xGlobSim);
	      hMap["all_GlobSimPos_y"] ->Fill(yGlobSim);
	      hMap["all_GlobSimPos_z"] ->Fill(zGlobSim);

	      hMap["all_GlobRecPos_x"] ->Fill( xGlobRec );
	      hMap["all_GlobRecPos_y"] ->Fill( yGlobRec);
	      hMap["all_GlobRecPos_z"] ->Fill( zGlobRec);



}





void FastTrackAnalyzer::endJob(){

  TFile* outfile = new TFile("ResHistos.root", "RECREATE");
  outfile->cd();
  for(std::map<TString, TH1F*>::iterator it = hMap.begin(); it!=hMap.end(); it++){
    it->second->Write();
  }
  outfile->Close();
}

//define this as a plug-in
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(FastTrackAnalyzer);

