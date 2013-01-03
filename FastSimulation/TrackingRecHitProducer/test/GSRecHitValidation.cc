//Alexander.Schmidt@cern.ch
//March 2007

//Updated by Douglas.Orbaker@cern.ch
//January 2009

//This code produces histograms to compare FastSim MatchedRecHits to FullSim RecHits.

// Numbering scheme
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FastSimulation/TrackingRecHitProducer/test/GSRecHitValidation.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "TH1F.h"
#include "TFile.h"

using namespace edm;
using namespace std;
    
GSRecHitValidation::GSRecHitValidation(edm::ParameterSet const& conf) : 
  conf_(conf),
  matchedHitCollectionInputTag_(conf.getParameter<edm::InputTag>("matchedHitCollectionInputTag")),
  hitCollectionInputTag_(conf.getParameter<edm::InputTag>("hitCollectionInputTag")) {
  
  iEventCounter=0;
  
  trackerContainers.clear();
  trackerContainers = conf.getParameter<std::vector<std::string> >("SimHitList");

  // histogram axis limits for RecHit validation plots
  //PXB
  PXB_Res_AxisLim =      conf.getParameter<double>("PXB_Res_AxisLim" );  
  PXB_RecPos_AxisLim =   conf.getParameter<double>("PXB_RecPos_AxisLim" );
  PXB_SimPos_AxisLim =   conf.getParameter<double>("PXB_SimPos_AxisLim" );
  PXB_Err_AxisLim =      conf.getParameter<double>("PXB_Err_AxisLim");
  
  //PXF
  PXF_Res_AxisLim =      conf.getParameter<double>("PXF_Res_AxisLim" );
  PXF_RecPos_AxisLim =   conf.getParameter<double>("PXF_RecPos_AxisLim" );
  PXF_SimPos_AxisLim =   conf.getParameter<double>("PXF_SimPos_AxisLim" );
  PXF_Err_AxisLim =      conf.getParameter<double>("PXF_Err_AxisLim");
  
  //TIB
  TIB_Pos_x_AxisLim =     conf.getParameter<double>("TIB_Pos_x_AxisLim" );
  TIB_Pos_y_AxisLim =     conf.getParameter<double>("TIB_Pos_y_AxisLim" );
  TIB_Res_x_AxisLim =     conf.getParameter<double>("TIB_Res_x_AxisLim" );
  TIB_Res_y_AxisLim =     conf.getParameter<double>("TIB_Res_y_AxisLim" );
  TIB_Pull_x_AxisLim =    conf.getParameter<double>("TIB_Pull_x_AxisLim" );
  TIB_Pull_y_AxisLim =    conf.getParameter<double>("TIB_Pull_y_AxisLim" );
  
  //TOB
  TOB_Pos_x_AxisLim =     conf.getParameter<double>("TOB_Pos_x_AxisLim" );
  TOB_Pos_y_AxisLim =     conf.getParameter<double>("TOB_Pos_y_AxisLim" );
  TOB_Res_x_AxisLim =     conf.getParameter<double>("TOB_Res_x_AxisLim" );
  TOB_Res_y_AxisLim =     conf.getParameter<double>("TOB_Res_y_AxisLim" );
  TOB_Pull_x_AxisLim =    conf.getParameter<double>("TOB_Pull_x_AxisLim" );
  TOB_Pull_y_AxisLim =    conf.getParameter<double>("TOB_Pull_y_AxisLim" );
  
  //TID
  TID_Pos_x_AxisLim =     conf.getParameter<double>("TID_Pos_x_AxisLim" );
  TID_Pos_y_AxisLim =     conf.getParameter<double>("TID_Pos_y_AxisLim" );
  TID_Res_x_AxisLim =     conf.getParameter<double>("TID_Res_x_AxisLim" );
  TID_Res_y_AxisLim =     conf.getParameter<double>("TID_Res_y_AxisLim" );
  TID_Pull_x_AxisLim =    conf.getParameter<double>("TID_Pull_x_AxisLim" );
  TID_Pull_y_AxisLim =    conf.getParameter<double>("TID_Pull_y_AxisLim" );

  //TEC
  TEC_Pos_x_AxisLim =     conf.getParameter<double>("TEC_Pos_x_AxisLim" );
  TEC_Pos_y_AxisLim =     conf.getParameter<double>("TEC_Pos_y_AxisLim" );
  TEC_Res_x_AxisLim =     conf.getParameter<double>("TEC_Res_x_AxisLim" );
  TEC_Res_y_AxisLim =     conf.getParameter<double>("TEC_Res_y_AxisLim" );
  TEC_Pull_x_AxisLim =    conf.getParameter<double>("TEC_Pull_x_AxisLim" );
  TEC_Pull_y_AxisLim =    conf.getParameter<double>("TEC_Pull_y_AxisLim" );
  
  outfilename = conf.getParameter<string>("outfilename");
}//end constructor



GSRecHitValidation::~GSRecHitValidation() {}

void GSRecHitValidation::beginJob(){
  // book histograms  for plots
  // number of pixel hits
  hMap["NumSimPixHits"] = new TH1F("NumSimPixHits", "NumSimPixHits", 100, 0,10);
  hMap["NumRecPixHits"] = new TH1F("NumRecPixHits", "NumRecPixHits", 100, 0,10);

  //number of strip hits
  hMap["NumSimStripHits"] = new TH1F("NumSimStripHits", "NumSimStripHits", 100, 0,30);
  hMap["NumRecStripHits"] = new TH1F("NumRecStripHits", "NumRecStripHits", 100, 0,30);

  // number of total hits
  hMap["NumSimHits"] = new TH1F("NumSimHits", "NumSimHits", 100, 0,35);
  hMap["NumRecHits"] = new TH1F("NumRecHits", "NumRecHits", 100, 0,35);

  // global position of all hits
  hMap["GlobSimPos_x"] = new TH1F("GlobSimPos_x","GlobSimPos_x" ,300, 0,115);
  hMap["GlobSimPos_y"] = new TH1F("GlobSimPos_y","GlobSimPos_y" ,300, 0,115);
  hMap["GlobSimPos_z"] = new TH1F("GlobSimPos_z","GlobSimPos_z" ,300, 0,300);

  hMap["GlobRecPos_x"] = new TH1F("GlobRecPos_x","GlobRecPos_x" ,300, 0,115);
  hMap["GlobRecPos_y"] = new TH1F("GlobRecPos_y","GlobRecPos_y" ,300, 0,115);
  hMap["GlobRecPos_z"] = new TH1F("GlobRecPos_z","GlobRecPos_z" ,300, 0,300);
  
  
  // strip maximum 7 layers (TEC)
  //include pixels too
  for(int i=1; i<=7; i++){
    TString index = ""; index+=i;
    //Layers 1+2
    //all pixel
    //all glued -> rphi, sas and matched for all
    if (i < 3) {
      //PXB
      hMap["PXB_Res_x_"+index] =    new TH1F("PXB_Res_x_"+index,    "PXB_Res_x_"+index,    100,  -PXB_Res_AxisLim,    PXB_Res_AxisLim);
      hMap["PXB_Res_y_"+index] =    new TH1F("PXB_Res_y_"+index,    "PXB_Res_y_"+index,    100,  -PXB_Res_AxisLim,    PXB_Res_AxisLim);
      hMap["PXB_SimPos_x_"+index] = new TH1F("PXB_SimPos_x_"+index, "PXB_SimPos_x_"+index, 100,  -PXB_SimPos_AxisLim, PXB_SimPos_AxisLim);
      hMap["PXB_SimPos_y_"+index] = new TH1F("PXB_SimPos_y_"+index, "PXB_SimPos_y_"+index, 100,  -PXB_SimPos_AxisLim, PXB_SimPos_AxisLim);
      hMap["PXB_RecPos_x_"+index] = new TH1F("PXB_RecPos_x_"+index, "PXB_RecPos_x_"+index, 100,  -PXB_RecPos_AxisLim, PXB_RecPos_AxisLim);
      hMap["PXB_RecPos_y_"+index] = new TH1F("PXB_RecPos_y_"+index, "PXB_RecPos_y_"+index, 100,  -PXB_RecPos_AxisLim, PXB_RecPos_AxisLim);
      hMap["PXB_Err_xx"+index] =    new TH1F("PXB_Err_xx"+index, "   PXB_Err_xx"+index,    100,  -PXB_Err_AxisLim,    PXB_Err_AxisLim);
      hMap["PXB_Err_xy"+index] =    new TH1F("PXB_Err_xy"+index,    "PXB_Err_xy"+index,    100,  -PXB_Err_AxisLim,    PXB_Err_AxisLim);
      hMap["PXB_Err_yy"+index] =    new TH1F("PXB_Err_yy"+index,    "PXB_Err_yy"+index,    100,  -PXB_Err_AxisLim,    PXB_Err_AxisLim);
      
      //PXF
      hMap["PXF_Res_x_"+index] =    new TH1F("PXF_Res_x_"+index,    "PXF_Res_x_"+index,    100,  -PXF_Res_AxisLim,    PXF_Res_AxisLim);
      hMap["PXF_Res_y_"+index] =    new TH1F("PXF_Res_y_"+index,    "PXF_Res_y_"+index,    100,  -PXF_Res_AxisLim,    PXF_Res_AxisLim);
      hMap["PXF_SimPos_x_"+index] = new TH1F("PXF_SimPos_x_"+index, "PXF_SimPos_x_"+index, 100,  -PXF_SimPos_AxisLim, PXF_SimPos_AxisLim);
      hMap["PXF_SimPos_y_"+index] = new TH1F("PXF_SimPos_y_"+index, "PXF_SimPos_y_"+index, 100,  -PXF_SimPos_AxisLim, PXF_SimPos_AxisLim);
      hMap["PXF_RecPos_x_"+index] = new TH1F("PXF_RecPos_x_"+index, "PXF_RecPos_x_"+index, 100,  -PXF_RecPos_AxisLim, PXF_RecPos_AxisLim);
      hMap["PXF_RecPos_y_"+index] = new TH1F("PXF_RecPos_y_"+index, "PXF_RecPos_y_"+index, 100,  -PXF_RecPos_AxisLim, PXF_RecPos_AxisLim);
      hMap["PXF_Err_xx"+index] =    new TH1F("PXF_Err_xx"+index, "   PXF_Err_xx"+index,    100,  -PXF_Err_AxisLim,    PXF_Err_AxisLim);
      hMap["PXF_Err_xy"+index] =    new TH1F("PXF_Err_xy"+index,    "PXF_Err_xy"+index,    100,  -PXF_Err_AxisLim,    PXF_Err_AxisLim);
      hMap["PXF_Err_yy"+index] =    new TH1F("PXF_Err_yy"+index,    "PXF_Err_yy"+index,    100,  -PXF_Err_AxisLim,    PXF_Err_AxisLim);
 
      //TIB
      hMap["TIB_rphi_Res_x_"+index] =     new TH1F("TIB_rphi_Res_x_"+index,     "TIB_rphi_Res_x_"+index,     100, -TIB_Res_x_AxisLim,     TIB_Res_x_AxisLim);
      hMap["TIB_sas_Res_x_"+index] =      new TH1F("TIB_sas_Res_x_"+index,      "TIB_sas_Res_x_"+index,      100, -TIB_Res_x_AxisLim,     TIB_Res_x_AxisLim);
      hMap["TIB_matched_Res_x_"+index] =  new TH1F("TIB_matched_Res_x_"+index,  "TIB_matched_Res_x_"+index,  100, -TIB_Res_x_AxisLim,     TIB_Res_x_AxisLim);
      hMap["TIB_matched_Res_y_"+index] =  new TH1F("TIB_matched_Res_y_"+index,  "TIB_matched_Res_y_"+index,  100, -TIB_Res_y_AxisLim,     TIB_Res_y_AxisLim);
      hMap["TIB_rphi_Pos_x_"+index] =     new TH1F("TIB_rphi_Pos_x_"+index,     "TIB_rphi_Pos_x_"+index,     100, -TIB_Pos_x_AxisLim,     TIB_Pos_x_AxisLim);
      hMap["TIB_sas_Pos_x_"+index] =      new TH1F("TIB_sas_Pos_x_"+index,      "TIB_sas_Pos_x_"+index,      100, -TIB_Pos_x_AxisLim,     TIB_Pos_x_AxisLim);
      hMap["TIB_matched_Pos_x_"+index] =  new TH1F("TIB_matched_Pos_x_"+index,  "TIB_matched_Pos_x_"+index,  100, -TIB_Pos_x_AxisLim,     TIB_Pos_x_AxisLim);
      hMap["TIB_matched_Pos_y_"+index] =  new TH1F("TIB_matched_Pos_y_"+index,  "TIB_matched_Pos_y_"+index,  100, -TIB_Pos_y_AxisLim,     TIB_Pos_y_AxisLim);
      hMap["TIB_rphi_Pull_x_"+index] =    new TH1F("TIB_rphi_Pull_x_"+index,    "TIB_rphi_Pull_x_"+index,    100, -TIB_Pull_x_AxisLim,    TIB_Pull_x_AxisLim);
      hMap["TIB_sas_Pull_x_"+index] =     new TH1F("TIB_sas_Pull_x_"+index,     "TIB_sas_Pull_x_"+index,     100, -TIB_Pull_x_AxisLim,    TIB_Pull_x_AxisLim);
      hMap["TIB_matched_Pull_x_"+index] = new TH1F("TIB_matched_Pull_x_"+index, "TIB_matched_Pull_x_"+index, 100, -TIB_Pull_x_AxisLim,    TIB_Pull_x_AxisLim);
      hMap["TIB_matched_Pull_y_"+index] = new TH1F("TIB_matched_Pull_y_"+index, "TIB_matched_Pull_y_"+index, 100, -TIB_Pull_y_AxisLim,    TIB_Pull_y_AxisLim);
      
      //TOB
      hMap["TOB_rphi_Res_x_"+index] =     new TH1F("TOB_rphi_Res_x_"+index,     "TOB_rphi_Res_x_"+index,     100, -TOB_Res_x_AxisLim,     TOB_Res_x_AxisLim);
      hMap["TOB_sas_Res_x_"+index] =      new TH1F("TOB_sas_Res_x_"+index,      "TOB_sas_Res_x_"+index,      100, -TOB_Res_x_AxisLim,     TOB_Res_x_AxisLim);
      hMap["TOB_matched_Res_x_"+index] =  new TH1F("TOB_matched_Res_x_"+index,  "TOB_matched_Res_x_"+index,  100, -TOB_Res_x_AxisLim,     TOB_Res_x_AxisLim);
      hMap["TOB_matched_Res_y_"+index] =  new TH1F("TOB_matched_Res_y_"+index,  "TOB_matched_Res_y_"+index,  100, -TOB_Res_y_AxisLim,     TOB_Res_y_AxisLim);
      hMap["TOB_rphi_Pos_x_"+index] =     new TH1F("TOB_rphi_Pos_x_"+index,     "TOB_rphi_Pos_x_"+index,     100, -TOB_Pos_x_AxisLim,     TOB_Pos_x_AxisLim);
      hMap["TOB_sas_Pos_x_"+index] =      new TH1F("TOB_sas_Pos_x_"+index,      "TOB_sas_Pos_x_"+index,      100, -TOB_Pos_x_AxisLim,     TOB_Pos_x_AxisLim);
      hMap["TOB_matched_Pos_x_"+index] =  new TH1F("TOB_matched_Pos_x_"+index,  "TOB_matched_Pos_x_"+index,  100, -TOB_Pos_x_AxisLim,     TOB_Pos_x_AxisLim);
      hMap["TOB_matched_Pos_y_"+index] =  new TH1F("TOB_matched_Pos_y_"+index,  "TOB_matched_Pos_y_"+index,  100, -TOB_Pos_y_AxisLim,     TOB_Pos_y_AxisLim);
      hMap["TOB_rphi_Pull_x_"+index] =    new TH1F("TOB_rphi_Pull_x_"+index,    "TOB_rphi_Pull_x_"+index,    100, -TOB_Pull_x_AxisLim,    TOB_Pull_x_AxisLim);
      hMap["TOB_sas_Pull_x_"+index] =     new TH1F("TOB_sas_Pull_x_"+index,     "TOB_sas_Pull_x_"+index,     100, -TOB_Pull_x_AxisLim,    TOB_Pull_x_AxisLim);
      hMap["TOB_matched_Pull_x_"+index] = new TH1F("TOB_matched_Pull_x_"+index, "TOB_matched_Pull_x_"+index, 100, -TOB_Pull_x_AxisLim,    TOB_Pull_x_AxisLim);
      hMap["TOB_matched_Pull_y_"+index] = new TH1F("TOB_matched_Pull_y_"+index, "TOB_matched_Pull_y_"+index, 100, -TOB_Pull_y_AxisLim,    TOB_Pull_y_AxisLim);
      
      //TID
      hMap["TID_rphi_Res_x_"+index] =     new TH1F("TID_rphi_Res_x_"+index,     "TID_rphi_Res_x_"+index,     100, -TID_Res_x_AxisLim,     TID_Res_x_AxisLim);
      hMap["TID_sas_Res_x_"+index] =      new TH1F("TID_sas_Res_x_"+index,      "TID_sas_Res_x_"+index,      100, -TID_Res_x_AxisLim,     TID_Res_x_AxisLim);
      hMap["TID_matched_Res_x_"+index] =  new TH1F("TID_matched_Res_x_"+index,  "TID_matched_Res_x_"+index,  100, -TID_Res_x_AxisLim,     TID_Res_x_AxisLim);
      hMap["TID_matched_Res_y_"+index] =  new TH1F("TID_matched_Res_y_"+index,  "TID_matched_Res_y_"+index,  100, -TID_Res_y_AxisLim,     TID_Res_y_AxisLim);
      hMap["TID_rphi_Pos_x_"+index] =     new TH1F("TID_rphi_Pos_x_"+index,     "TID_rphi_Pos_x_"+index,     100, -TID_Pos_x_AxisLim,     TID_Pos_x_AxisLim);
      hMap["TID_sas_Pos_x_"+index] =      new TH1F("TID_sas_Pos_x_"+index,      "TID_sas_Pos_x_"+index,      100, -TID_Pos_x_AxisLim,     TID_Pos_x_AxisLim);
      hMap["TID_matched_Pos_x_"+index] =  new TH1F("TID_matched_Pos_x_"+index,  "TID_matched_Pos_x_"+index,  100, -TID_Pos_x_AxisLim,     TID_Pos_x_AxisLim);
      hMap["TID_matched_Pos_y_"+index] =  new TH1F("TID_matched_Pos_y_"+index,  "TID_matched_Pos_y_"+index,  100, -TID_Pos_y_AxisLim,     TID_Pos_y_AxisLim);
      hMap["TID_rphi_Pull_x_"+index] =    new TH1F("TID_rphi_Pull_x_"+index,    "TID_rphi_Pull_x_"+index,    100, -TID_Pull_x_AxisLim,    TID_Pull_x_AxisLim);
      hMap["TID_sas_Pull_x_"+index] =     new TH1F("TID_sas_Pull_x_"+index,     "TID_sas_Pull_x_"+index,     100, -TID_Pull_x_AxisLim,    TID_Pull_x_AxisLim);
      hMap["TID_matched_Pull_x_"+index] = new TH1F("TID_matched_Pull_x_"+index, "TID_matched_Pull_x_"+index, 100, -TID_Pull_x_AxisLim,    TID_Pull_x_AxisLim);
      hMap["TID_matched_Pull_y_"+index] = new TH1F("TID_matched_Pull_y_"+index, "TID_matched_Pull_y_"+index, 100, -TID_Pull_y_AxisLim,    TID_Pull_y_AxisLim);
    
      //TEC
      hMap["TEC_rphi_Res_x_"+index] =     new TH1F("TEC_rphi_Res_x_"+index,     "TEC_rphi_Res_x_"+index,     100, -TEC_Res_x_AxisLim,     TEC_Res_x_AxisLim);
      hMap["TEC_sas_Res_x_"+index] =      new TH1F("TEC_sas_Res_x_"+index,      "TEC_sas_Res_x_"+index,      100, -TEC_Res_x_AxisLim,     TEC_Res_x_AxisLim);
      hMap["TEC_matched_Res_x_"+index] =  new TH1F("TEC_matched_Res_x_"+index,  "TEC_matched_Res_x_"+index,  100, -TEC_Res_x_AxisLim,     TEC_Res_x_AxisLim);
      hMap["TEC_matched_Res_y_"+index] =  new TH1F("TEC_matched_Res_y_"+index,  "TEC_matched_Res_y_"+index,  100, -TEC_Res_y_AxisLim,     TEC_Res_y_AxisLim);
      hMap["TEC_rphi_Pos_x_"+index] =     new TH1F("TEC_rphi_Pos_x_"+index,     "TEC_rphi_Pos_x_"+index,     100, -TEC_Pos_x_AxisLim,     TEC_Pos_x_AxisLim);
      hMap["TEC_sas_Pos_x_"+index] =      new TH1F("TEC_sas_Pos_x_"+index,      "TEC_sas_Pos_x_"+index,      100, -TEC_Pos_x_AxisLim,     TEC_Pos_x_AxisLim);
      hMap["TEC_matched_Pos_x_"+index] =  new TH1F("TEC_matched_Pos_x_"+index,  "TEC_matched_Pos_x_"+index,  100, -TEC_Pos_x_AxisLim,     TEC_Pos_x_AxisLim);
      hMap["TEC_matched_Pos_y_"+index] =  new TH1F("TEC_matched_Pos_y_"+index,  "TEC_matched_Pos_y_"+index,  100, -TEC_Pos_y_AxisLim,     TEC_Pos_y_AxisLim);
      hMap["TEC_rphi_Pull_x_"+index] =    new TH1F("TEC_rphi_Pull_x_"+index,    "TEC_rphi_Pull_x_"+index,    100, -TEC_Pull_x_AxisLim,    TEC_Pull_x_AxisLim);
      hMap["TEC_sas_Pull_x_"+index] =     new TH1F("TEC_sas_Pull_x_"+index,     "TEC_sas_Pull_x_"+index,     100, -TEC_Pull_x_AxisLim,    TEC_Pull_x_AxisLim);
      hMap["TEC_matched_Pull_x_"+index] = new TH1F("TEC_matched_Pull_x_"+index, "TEC_matched_Pull_x_"+index, 100, -TEC_Pull_x_AxisLim,    TEC_Pull_x_AxisLim);
      hMap["TEC_matched_Pull_y_"+index] = new TH1F("TEC_matched_Pull_y_"+index, "TEC_matched_Pull_y_"+index, 100, -TEC_Pull_y_AxisLim,    TEC_Pull_y_AxisLim);
    }
    
    //Layer 3
    //PXF gone, only PXB
    //rphi single sided for all
    if(i==3){
      //PXB
      hMap["PXB_Res_x_"+index] =    new TH1F("PXB_Res_x_"+index,    "PXB_Res_x_"+index,    100,  -PXB_Res_AxisLim,    PXB_Res_AxisLim);
      hMap["PXB_Res_y_"+index] =    new TH1F("PXB_Res_y_"+index,    "PXB_Res_y_"+index,    100,  -PXB_Res_AxisLim,    PXB_Res_AxisLim);
      hMap["PXB_SimPos_x_"+index] = new TH1F("PXB_SimPos_x_"+index, "PXB_SimPos_x_"+index, 100,  -PXB_SimPos_AxisLim, PXB_SimPos_AxisLim);
      hMap["PXB_SimPos_y_"+index] = new TH1F("PXB_SimPos_y_"+index, "PXB_SimPos_y_"+index, 100,  -PXB_SimPos_AxisLim, PXB_SimPos_AxisLim);
      hMap["PXB_RecPos_x_"+index] = new TH1F("PXB_RecPos_x_"+index, "PXB_RecPos_x_"+index, 100,  -PXB_RecPos_AxisLim, PXB_RecPos_AxisLim);
      hMap["PXB_RecPos_y_"+index] = new TH1F("PXB_RecPos_y_"+index, "PXB_RecPos_y_"+index, 100,  -PXB_RecPos_AxisLim, PXB_RecPos_AxisLim);
      hMap["PXB_Err_xx"+index] =    new TH1F("PXB_Err_xx"+index, "   PXB_Err_xx"+index,    100,  -PXB_Err_AxisLim,    PXB_Err_AxisLim);
      hMap["PXB_Err_xy"+index] =    new TH1F("PXB_Err_xy"+index,    "PXB_Err_xy"+index,    100,  -PXB_Err_AxisLim,    PXB_Err_AxisLim);
      hMap["PXB_Err_yy"+index] =    new TH1F("PXB_Err_yy"+index,    "PXB_Err_yy"+index,    100,  -PXB_Err_AxisLim,    PXB_Err_AxisLim);
      
      //TIB
      hMap["TIB_rphi_Res_x_"+index] =     new TH1F("TIB_rphi_Res_x_"+index,     "TIB_rphi_Res_x_"+index,     100, -TIB_Res_x_AxisLim,     TIB_Res_x_AxisLim);
      hMap["TIB_rphi_Pos_x_"+index] =     new TH1F("TIB_rphi_Pos_x_"+index,     "TIB_rphi_Pos_x_"+index,     100, -TIB_Pos_x_AxisLim,     TIB_Pos_x_AxisLim);
      hMap["TIB_rphi_Pull_x_"+index] =    new TH1F("TIB_rphi_Pull_x_"+index,    "TIB_rphi_Pull_x_"+index,    100, -TIB_Pull_x_AxisLim,    TIB_Pull_x_AxisLim);
      
      //TOB
      hMap["TOB_rphi_Res_x_"+index] =     new TH1F("TOB_rphi_Res_x_"+index,     "TOB_rphi_Res_x_"+index,     100, -TOB_Res_x_AxisLim,     TOB_Res_x_AxisLim);
      hMap["TOB_rphi_Pos_x_"+index] =     new TH1F("TOB_rphi_Pos_x_"+index,     "TOB_rphi_Pos_x_"+index,     100, -TOB_Pos_x_AxisLim,     TOB_Pos_x_AxisLim);
      hMap["TOB_rphi_Pull_x_"+index] =    new TH1F("TOB_rphi_Pull_x_"+index,    "TOB_rphi_Pull_x_"+index,    100, -TOB_Pull_x_AxisLim,    TOB_Pull_x_AxisLim);
      
      //TID
      hMap["TID_rphi_Res_x_"+index] =     new TH1F("TID_rphi_Res_x_"+index,     "TID_rphi_Res_x_"+index,     100, -TID_Res_x_AxisLim,     TID_Res_x_AxisLim);
      hMap["TID_rphi_Pos_x_"+index] =     new TH1F("TID_rphi_Pos_x_"+index,     "TID_rphi_Pos_x_"+index,     100, -TID_Pos_x_AxisLim,     TID_Pos_x_AxisLim);
      hMap["TID_rphi_Pull_x_"+index] =    new TH1F("TID_rphi_Pull_x_"+index,    "TID_rphi_Pull_x_"+index,    100, -TID_Pull_x_AxisLim,    TID_Pull_x_AxisLim);
      
      //TEC
      hMap["TEC_rphi_Res_x_"+index] =     new TH1F("TEC_rphi_Res_x_"+index,     "TEC_rphi_Res_x_"+index,     100, -TEC_Res_x_AxisLim,     TEC_Res_x_AxisLim);
      hMap["TEC_rphi_Pos_x_"+index] =     new TH1F("TEC_rphi_Pos_x_"+index,     "TEC_rphi_Pos_x_"+index,     100, -TEC_Pos_x_AxisLim,     TEC_Pos_x_AxisLim);
      hMap["TEC_rphi_Pull_x_"+index] =    new TH1F("TEC_rphi_Pull_x_"+index,    "TEC_rphi_Pull_x_"+index,    100, -TEC_Pull_x_AxisLim,    TEC_Pull_x_AxisLim);
    }

    //Layer 4
    //no pixels
    //TID gone, other rphi single sided
    if(i==4){
      //TIB
      hMap["TIB_rphi_Res_x_"+index] =     new TH1F("TIB_rphi_Res_x_"+index,     "TIB_rphi_Res_x_"+index,     100, -TIB_Res_x_AxisLim,     TIB_Res_x_AxisLim);
      hMap["TIB_rphi_Pos_x_"+index] =     new TH1F("TIB_rphi_Pos_x_"+index,     "TIB_rphi_Pos_x_"+index,     100, -TIB_Pos_x_AxisLim,     TIB_Pos_x_AxisLim);
      hMap["TIB_rphi_Pull_x_"+index] =    new TH1F("TIB_rphi_Pull_x_"+index,    "TIB_rphi_Pull_x_"+index,    100, -TIB_Pull_x_AxisLim,    TIB_Pull_x_AxisLim);
      
      //TOB
      hMap["TOB_rphi_Res_x_"+index] =     new TH1F("TOB_rphi_Res_x_"+index,     "TOB_rphi_Res_x_"+index,     100, -TOB_Res_x_AxisLim,     TOB_Res_x_AxisLim);
      hMap["TOB_rphi_Pos_x_"+index] =     new TH1F("TOB_rphi_Pos_x_"+index,     "TOB_rphi_Pos_x_"+index,     100, -TOB_Pos_x_AxisLim,     TOB_Pos_x_AxisLim);
      hMap["TOB_rphi_Pull_x_"+index] =    new TH1F("TOB_rphi_Pull_x_"+index,    "TOB_rphi_Pull_x_"+index,    100, -TOB_Pull_x_AxisLim,    TOB_Pull_x_AxisLim);
      
      //TEC
      hMap["TEC_rphi_Res_x_"+index] =     new TH1F("TEC_rphi_Res_x_"+index,     "TEC_rphi_Res_x_"+index,     100, -TEC_Res_x_AxisLim,     TEC_Res_x_AxisLim);
      hMap["TEC_rphi_Pos_x_"+index] =     new TH1F("TEC_rphi_Pos_x_"+index,     "TEC_rphi_Pos_x_"+index,     100, -TEC_Pos_x_AxisLim,     TEC_Pos_x_AxisLim);
      hMap["TEC_rphi_Pull_x_"+index] =    new TH1F("TEC_rphi_Pull_x_"+index,    "TEC_rphi_Pull_x_"+index,    100, -TEC_Pull_x_AxisLim,    TEC_Pull_x_AxisLim);
    }
    
    //Layer 5
    //no pixels
    //TID + TIB gone, TOB rphi single sided, TEC glued -> rphi, sas and matched
    if(i==5){
      //TOB
      hMap["TOB_rphi_Res_x_"+index] =     new TH1F("TOB_rphi_Res_x_"+index,     "TOB_rphi_Res_x_"+index,     100, -TOB_Res_x_AxisLim,     TOB_Res_x_AxisLim);
      hMap["TOB_rphi_Pos_x_"+index] =     new TH1F("TOB_rphi_Pos_x_"+index,     "TOB_rphi_Pos_x_"+index,     100, -TOB_Pos_x_AxisLim,     TOB_Pos_x_AxisLim);
      hMap["TOB_rphi_Pull_x_"+index] =    new TH1F("TOB_rphi_Pull_x_"+index,    "TOB_rphi_Pull_x_"+index,    100, -TOB_Pull_x_AxisLim,    TOB_Pull_x_AxisLim);
      
      //TEC
      hMap["TEC_rphi_Res_x_"+index] =     new TH1F("TEC_rphi_Res_x_"+index,     "TEC_rphi_Res_x_"+index,     100, -TEC_Res_x_AxisLim,     TEC_Res_x_AxisLim);
      hMap["TEC_sas_Res_x_"+index] =      new TH1F("TEC_sas_Res_x_"+index,      "TEC_sas_Res_x_"+index,      100, -TEC_Res_x_AxisLim,     TEC_Res_x_AxisLim);
      hMap["TEC_matched_Res_x_"+index] =  new TH1F("TEC_matched_Res_x_"+index,  "TEC_matched_Res_x_"+index,  100, -TEC_Res_x_AxisLim,     TEC_Res_x_AxisLim);
      hMap["TEC_matched_Res_y_"+index] =  new TH1F("TEC_matched_Res_y_"+index,  "TEC_matched_Res_y_"+index,  100, -TEC_Res_y_AxisLim,     TEC_Res_y_AxisLim);
      hMap["TEC_rphi_Pos_x_"+index] =     new TH1F("TEC_rphi_Pos_x_"+index,     "TEC_rphi_Pos_x_"+index,     100, -TEC_Pos_x_AxisLim,     TEC_Pos_x_AxisLim);
      hMap["TEC_sas_Pos_x_"+index] =      new TH1F("TEC_sas_Pos_x_"+index,      "TEC_sas_Pos_x_"+index,      100, -TEC_Pos_x_AxisLim,     TEC_Pos_x_AxisLim);
      hMap["TEC_matched_Pos_x_"+index] =  new TH1F("TEC_matched_Pos_x_"+index,  "TEC_matched_Pos_x_"+index,  100, -TEC_Pos_x_AxisLim,     TEC_Pos_x_AxisLim);
      hMap["TEC_matched_Pos_y_"+index] =  new TH1F("TEC_matched_Pos_y_"+index,  "TEC_matched_Pos_y_"+index,  100, -TEC_Pos_y_AxisLim,     TEC_Pos_y_AxisLim);
      hMap["TEC_rphi_Pull_x_"+index] =    new TH1F("TEC_rphi_Pull_x_"+index,    "TEC_rphi_Pull_x_"+index,    100, -TEC_Pull_x_AxisLim,    TEC_Pull_x_AxisLim);
      hMap["TEC_sas_Pull_x_"+index] =     new TH1F("TEC_sas_Pull_x_"+index,     "TEC_sas_Pull_x_"+index,     100, -TEC_Pull_x_AxisLim,    TEC_Pull_x_AxisLim);
      hMap["TEC_matched_Pull_x_"+index] = new TH1F("TEC_matched_Pull_x_"+index, "TEC_matched_Pull_x_"+index, 100, -TEC_Pull_x_AxisLim,    TEC_Pull_x_AxisLim);
      hMap["TEC_matched_Pull_y_"+index] = new TH1F("TEC_matched_Pull_y_"+index, "TEC_matched_Pull_y_"+index, 100, -TEC_Pull_y_AxisLim,    TEC_Pull_y_AxisLim);
    }
    
    //Layer 6
    //no pixels
    //TID + TIB gone, TOB + TEC rphi single sided
    if(i==6){
      //TOB
      hMap["TOB_rphi_Res_x_"+index] =     new TH1F("TOB_rphi_Res_x_"+index,     "TOB_rphi_Res_x_"+index,     100, -TOB_Res_x_AxisLim,     TOB_Res_x_AxisLim);
      hMap["TOB_rphi_Pos_x_"+index] =     new TH1F("TOB_rphi_Pos_x_"+index,     "TOB_rphi_Pos_x_"+index,     100, -TOB_Pos_x_AxisLim,     TOB_Pos_x_AxisLim);
      hMap["TOB_rphi_Pull_x_"+index] =    new TH1F("TOB_rphi_Pull_x_"+index,    "TOB_rphi_Pull_x_"+index,    100, -TOB_Pull_x_AxisLim,    TOB_Pull_x_AxisLim);
      
      //TEC
      hMap["TEC_rphi_Res_x_"+index] =     new TH1F("TEC_rphi_Res_x_"+index,     "TEC_rphi_Res_x_"+index,     100, -TEC_Res_x_AxisLim,     TEC_Res_x_AxisLim);
      hMap["TEC_rphi_Pos_x_"+index] =     new TH1F("TEC_rphi_Pos_x_"+index,     "TEC_rphi_Pos_x_"+index,     100, -TEC_Pos_x_AxisLim,     TEC_Pos_x_AxisLim);
      hMap["TEC_rphi_Pull_x_"+index] =    new TH1F("TEC_rphi_Pull_x_"+index,    "TEC_rphi_Pull_x_"+index,    100, -TEC_Pull_x_AxisLim,    TEC_Pull_x_AxisLim);
    }
    
    //Layer 7
    //no pixels
    //TID, TIB + TOB gone, TOB rphi single sided
    if(i==7){
      //TEC
      hMap["TEC_rphi_Res_x_"+index] =     new TH1F("TEC_rphi_Res_x_"+index,     "TEC_rphi_Res_x_"+index,     100, -TEC_Res_x_AxisLim,     TEC_Res_x_AxisLim);
      hMap["TEC_rphi_Pos_x_"+index] =     new TH1F("TEC_rphi_Pos_x_"+index,     "TEC_rphi_Pos_x_"+index,     100, -TEC_Pos_x_AxisLim,     TEC_Pos_x_AxisLim);
      hMap["TEC_rphi_Pull_x_"+index] =    new TH1F("TEC_rphi_Pull_x_"+index,    "TEC_rphi_Pull_x_"+index,    100, -TEC_Pull_x_AxisLim,    TEC_Pull_x_AxisLim);
    }
  }//end layer loop
}//end begin job



void GSRecHitValidation::analyze(const edm::Event& event, const edm::EventSetup& setup)
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

  
  std::cout << "Event ID = "<< event.id() << std::endl ;
  
  // Get PSimHit's of the Event
  edm::Handle<CrossingFrame<PSimHit> > cf_simhit; 
  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
  for(uint32_t i=0; i<trackerContainers.size(); i++){
    event.getByLabel("mix",trackerContainers[i], cf_simhit);
    cf_simhitvec.push_back(cf_simhit.product());
  }
  
  std::auto_ptr<MixCollection<PSimHit> > allSimHits(new MixCollection<PSimHit>(cf_simhitvec));
  
  //Get RecHits from the event
  edm::Handle<SiTrackerGSMatchedRecHit2DCollection> theGSMatchedRecHits;
  event.getByLabel(matchedHitCollectionInputTag_, theGSMatchedRecHits);

  //Get RecHits from the event
  edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
  event.getByLabel(hitCollectionInputTag_, theGSRecHits);
  
  // stop with error if empty RecHit collection
  if(theGSMatchedRecHits->size() == 0) {
    std::cout<<" AS: theGSRecMatchedHits->size() == 0" << std::endl;
    return;
  }
  
  //counters for number of hits
  unsigned int iPixSimHits = 0;
  unsigned int iPixRecHits = 0;
  unsigned int iStripSimHits=0;
  unsigned int iStripRecHits=0;
  unsigned int iTotalRecHits = 0;
  unsigned int iTotalSimHits = 0;
    
  //loop on all rechits, match them to simhits and make plots
  for(edm::OwnVector<SiTrackerGSMatchedRecHit2D>::const_iterator iterrechit = theGSMatchedRecHits->begin();
      iterrechit != theGSMatchedRecHits->end(); ++iterrechit) { 
    
    // count rechits
    unsigned int subdet = iterrechit->geographicalId().subdetId();
    iTotalRecHits++;
    if(subdet==1 || subdet==2) iPixRecHits++;
    else if(subdet==3|| subdet==4 || subdet==5 || subdet == 6) iStripRecHits++;
    else { // for debugging
      std::cout<<" AS: ERROR simhit subdet inconsistent"<< std::endl;
      exit(1);
    }
    
    //match sim hit to rec hit
    unsigned int matchedSimHits = 0;
    PSimHit* simHit = NULL;
    
    for (MixCollection<PSimHit>::iterator isim=allSimHits->begin(); isim!= allSimHits->end(); isim++) {
      
      if(iterrechit->isMatched()) {
	//mono
	if(iterrechit->monoHit()->geographicalId().rawId() == isim->detUnitId()) {
	  if((int)iterrechit->monoHit()->simtrackId() == (int)isim->trackId()) {
	    cout << "Mono Hit." << endl;
	    matchedSimHits++;
	    simHit = const_cast<PSimHit*>(&(*isim));  
	    fillHitsPlots("", &(*iterrechit), simHit,tTopo);
	  }
	}
	//stereo
	if(iterrechit->stereoHit()->geographicalId().rawId() == isim->detUnitId()) {
	  if((int)iterrechit->stereoHit()->simtrackId() == (int)isim->trackId()) {
	    cout << "Stereo Hit." << endl;
	    matchedSimHits++;
	    simHit = const_cast<PSimHit*>(&(*isim));  
	    fillHitsPlots("", &(*iterrechit), simHit,tTopo);
	  }
	}
      }
      
      //single
      else if(iterrechit->geographicalId().rawId() == isim->detUnitId()) { 
	if((int)iterrechit->simtrackId()== (int) isim->trackId()) {
	  cout << "Single Hit." << endl;
	  matchedSimHits++;
	  simHit = const_cast<PSimHit*>(&(*isim));  
	  fillHitsPlots("", &(*iterrechit), simHit,tTopo);
	}
      }
      
    }//sim hit loop
    
    //if(matchedSimHits==0){ // for debugging
      //cout << "Matched: " << iterrechit->isMatched() << endl;
      //std::cout<<"ERROR: matchedSimHits!=1 " << std::endl;
      //std::cout<<"ERROR: matchedSimHits =  " << matchedSimHits<<  std::endl;
    //}
    
  }//end rec hit loop
  
  // number of pixel hits
  hMap["NumSimPixHits"]->Fill(iPixSimHits);
  hMap["NumRecPixHits"]->Fill(iPixRecHits);
  // number of strip hits
  hMap["NumSimStripHits"]->Fill(iStripSimHits);
  hMap["NumRecStripHits"]->Fill(iStripRecHits);
  // total number of hits
  hMap["NumRecHits"]->Fill(iTotalRecHits);
  hMap["NumSimHits"]->Fill(iTotalSimHits);
}//end analyze



void GSRecHitValidation::fillHitsPlots(TString prefix, const SiTrackerGSMatchedRecHit2D *rechit, PSimHit *simhit, const TrackerTopology *tTopo){
  
  //For knowing whether rechit is mono, stereo or single.
  int isMono = 0;
  int isStereo = 0;
  int isSingle = 0;

  //For matched
  if(rechit->isMatched()) {
 
    if(rechit->monoHit()->geographicalId().rawId() == simhit->detUnitId()) {
      isMono = 1;
    }    
    else if(rechit->stereoHit()->geographicalId().rawId() == simhit->detUnitId()) {
      isStereo = 1;
    }
    else {
      cout << "Split hit not matched to sim hit." << endl;
      exit(1);
    }
  
  }//end matched
  
  //For single sided
  else if (rechit->geographicalId().rawId() == simhit->detUnitId()) {
    isSingle = 1;
  }
  
  //for?
  else {
    cout << "Single sided hit not matched to sim hit." << endl;
    exit(1);
  }
  
  //Variables that depend on rec hit type.
  //Global position for rec hit
  float xGlobRec = 0;
  float yGlobRec = 0;
  float zGlobRec = 0;
  
  //Local position for rec hit
  float xRec = 0;
  float yRec = 0;
    
  //Local error for rec hit
  float Rec_err_xx = 0;
  float Rec_err_xy = 0;
  float Rec_err_yy = 0;

  //Getting the det unit
  const GeomDet* det = trackerG->idToDet( simhit->detUnitId() ); 
  
  //Global positions for the sim hits.
  GlobalPoint posGlobSim =  det->surface().toGlobal( simhit->localPosition()  );
  float xGlobSim = posGlobSim.x();
  float yGlobSim = posGlobSim.y();
  float zGlobSim = posGlobSim.z();
  
  //Global position for the rec hits (filled below).
  GlobalPoint posGlobRec;

  //mono
  if( isMono == 1 ) {
    //Global positions for the mono rec hits.
    posGlobRec =  det->surface().toGlobal( rechit->monoHit()->localPosition() );
    
    //Local positions for mono rec hit
    xRec = rechit->monoHit()->localPosition().x();
    yRec = rechit->monoHit()->localPosition().y();
    
    //Local errors for mono hit (Sqrt?)
    Rec_err_xx = sqrt(rechit->monoHit()->localPositionError().xx());
    Rec_err_xy = sqrt(rechit->monoHit()->localPositionError().xy());
    Rec_err_yy = sqrt(rechit->monoHit()->localPositionError().yy());
  }
  
  //stereo
  else if ( isStereo == 1 ) {
    //Global positions for the sim hits.
    posGlobRec =  det->surface().toGlobal( rechit->stereoHit()->localPosition() );
    
    //Local positions for stereo rec hit
    xRec = rechit->stereoHit()->localPosition().x();
    yRec = rechit->stereoHit()->localPosition().y();
    
    //Local errors for stereo hit (Sqrt?)
    Rec_err_xx = sqrt(rechit->stereoHit()->localPositionError().xx());
    Rec_err_xy = sqrt(rechit->stereoHit()->localPositionError().xy());
    Rec_err_yy = sqrt(rechit->stereoHit()->localPositionError().yy());
  }
  
  //single sided
  else if( isSingle == 1 ) {
    //Global positions for the single sided rec hits.
    posGlobRec =  det->surface().toGlobal( rechit->localPosition() );
    
    //Local positions for singled side rec hit
    xRec = rechit->localPosition().x();
    yRec = rechit->localPosition().y();
    
    //Local errors for single sided rec hit (Sqrt?)
    Rec_err_xx = sqrt(rechit->localPositionError().xx());
    Rec_err_xy = sqrt(rechit->localPositionError().xy());
    Rec_err_yy = sqrt(rechit->localPositionError().yy());
  }
  
  //???
  else {
    cout << "Hit somehow made it to this point without being matched to sim hit." << endl;
    exit(1);
  }
  
  //Global pos info for rec hit
  xGlobRec = posGlobRec.x();
  yGlobRec = posGlobRec.y();
  zGlobRec = posGlobRec.z();
  
  //Local information for the sim hit
  float xSim = simhit->localPosition().x();
  float ySim = simhit->localPosition().y();
  //  float zSim = simhit->localPosition().z();
  
  unsigned int subdet   = DetId(simhit->detUnitId()).subdetId();
  
  //Residuals for x and y
  float delta_x = xRec - xSim;
  float delta_y = yRec - ySim;
  
  //Pulls for x and y
  float pull_x = delta_x/Rec_err_xx;
  //  float pull_y = delta_y/Rec_err_yy;
  
  
  switch (subdet) {
    
    // PXB
  case 1: {
    
    unsigned int theLayer = tTopo->pxbLayer(simhit->detUnitId());
    TString layer = ""; layer+=theLayer;
    hMap[prefix+"PXB_RecPos_x_"+layer]->Fill(xRec);
    hMap[prefix+"PXB_RecPos_y_"+layer]->Fill(yRec);
    hMap[prefix+"PXB_SimPos_x_"+layer]->Fill(xSim);
    hMap[prefix+"PXB_SimPos_y_"+layer]->Fill(ySim);
    hMap[prefix+"PXB_Res_x_"+layer]->Fill(delta_x);
    hMap[prefix+"PXB_Res_y_"+layer]->Fill(delta_y);
    hMap[prefix+"PXB_Err_xx"+layer]->Fill(Rec_err_xx);
    hMap[prefix+"PXB_Err_xy"+layer]->Fill(Rec_err_xy);
    hMap[prefix+"PXB_Err_yy"+layer]->Fill(Rec_err_yy);
    break;
  }//case 1
    
    //PXF
  case 2:    {
    
    unsigned int theDisk = tTopo->pxfDisk(simhit->detUnitId());
    TString layer = ""; layer+=theDisk;
    hMap[prefix+"PXF_RecPos_x_"+layer]->Fill(xRec);
    hMap[prefix+"PXF_RecPos_y_"+layer]->Fill(yRec);
    hMap[prefix+"PXF_SimPos_x_"+layer]->Fill(xSim);
    hMap[prefix+"PXF_SimPos_y_"+layer]->Fill(ySim);
    hMap[prefix+"PXF_Res_x_"+layer]->Fill(delta_x);
    hMap[prefix+"PXF_Res_y_"+layer]->Fill(delta_y);
    hMap[prefix+"PXF_Err_xx"+layer]->Fill(Rec_err_xx);
    hMap[prefix+"PXF_Err_xy"+layer]->Fill(Rec_err_xy);
    hMap[prefix+"PXF_Err_yy"+layer]->Fill(Rec_err_yy);
    break;
  }//case 2
    
    
    // TIB
  case 3:
    {
      
      unsigned int theLayer  = tTopo->tibLayer(simhit->detUnitId());
      TString layer=""; layer+=theLayer;
      if ((isMono == 1)||(isSingle==1)) {
	hMap[prefix+"TIB_rphi_Pos_x_"+layer] ->Fill(xRec);
	hMap[prefix+"TIB_rphi_Res_x_"+layer] ->Fill(delta_x);
	hMap[prefix+"TIB_rphi_Pull_x_"+layer] ->Fill(pull_x);
      }
      else if (isStereo == 1) {
	hMap[prefix+"TIB_sas_Pos_x_"+layer] ->Fill(xRec);
	hMap[prefix+"TIB_sas_Res_x_"+layer] ->Fill(delta_x);
	hMap[prefix+"TIB_sas_Pull_x_"+layer] ->Fill(pull_x);
      }
      else {
	cout << "Oops TIB hit." << endl;
	exit(1);
      }

      break;
    }//case 3

    // TID
  case 4: 
    { 
      
      unsigned int theRing  = tTopo->tidRing(simhit->detUnitId());
      TString ring=""; ring+=theRing;
      
      if ((isMono == 1)||(isSingle==1)) {
	hMap[prefix+"TID_rphi_Pos_x_"+ring] ->Fill(xRec);
	hMap[prefix+"TID_rphi_Res_x_"+ring] ->Fill(delta_x);
	hMap[prefix+"TID_rphi_Pull_x_"+ring] ->Fill(pull_x);
      }
      else if (isStereo == 1) {
	hMap[prefix+"TID_sas_Pos_x_"+ring] ->Fill(xRec);
	hMap[prefix+"TID_sas_Res_x_"+ring] ->Fill(delta_x);
	hMap[prefix+"TID_sas_Pull_x_"+ring] ->Fill(pull_x);
      }
      else {
	cout << "Oops TID hit." << endl;
	exit(1);
      }

      break;
    }//case 4
    
    // TOB
  case 5:
    {
      
      unsigned int theLayer  = tTopo->tobLayer(simhit->detUnitId());
      TString layer=""; layer+=theLayer;
      
      if ((isMono == 1)||(isSingle==1)) {
	hMap[prefix+"TOB_rphi_Pos_x_"+layer] ->Fill(xRec);
	hMap[prefix+"TOB_rphi_Res_x_"+layer] ->Fill(delta_x);
	hMap[prefix+"TOB_rphi_Pull_x_"+layer] ->Fill(pull_x);
      }
      else if (isStereo == 1) {
	hMap[prefix+"TOB_sas_Pos_x_"+layer] ->Fill(xRec);
	hMap[prefix+"TOB_sas_Res_x_"+layer] ->Fill(delta_x);
	hMap[prefix+"TOB_sas_Pull_x_"+layer] ->Fill(pull_x);
      }
      else {
	cout << "Oops TOB hit." << endl;
	exit(1);
      }

      break;
    }//case 5

    // TEC
  case 6:
    {
      
      unsigned int theRing  = tTopo->tecRing(simhit->detUnitId());
      TString ring=""; ring+=theRing;
      
      if ((isMono == 1)||(isSingle==1)) {
	hMap[prefix+"TEC_rphi_Pos_x_"+ring] ->Fill(xRec);
	hMap[prefix+"TEC_rphi_Res_x_"+ring] ->Fill(delta_x);
	hMap[prefix+"TEC_rphi_Pull_x_"+ring] ->Fill(pull_x);
      }
      else if (isStereo == 1) {
	hMap[prefix+"TEC_sas_Pos_x_"+ring] ->Fill(xRec);
	hMap[prefix+"TEC_sas_Res_x_"+ring] ->Fill(delta_x);
	hMap[prefix+"TEC_sas_Pull_x_"+ring] ->Fill(pull_x);
      }
      else {
	cout << "Oops TEC hit." << endl;
	exit(1);
      }
      
      break;
    }//case 6
    
  }//end switch
  
  hMap["GlobSimPos_x"] ->Fill( xGlobSim );
  hMap["GlobSimPos_y"] ->Fill( yGlobSim );
  hMap["GlobSimPos_z"] ->Fill( zGlobSim );
  
  hMap["GlobRecPos_x"] ->Fill( xGlobRec );
  hMap["GlobRecPos_y"] ->Fill( yGlobRec );
  hMap["GlobRecPos_z"] ->Fill( zGlobRec );
}



void GSRecHitValidation::endJob(){
  
  TFile* outfile = new TFile(outfilename, "RECREATE");
  outfile->cd();
  for(std::map<TString, TH1F*>::iterator it = hMap.begin(); it!=hMap.end(); it++){
    it->second->Write();
  }
  outfile->Close();
  
}



//define this as a plug-in

DEFINE_FWK_MODULE(GSRecHitValidation);

