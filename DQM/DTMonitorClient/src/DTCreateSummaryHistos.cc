
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/11/24 09:17:30 $
 *  $Revision: 1.6 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTCreateSummaryHistos.h>

// Framework
#include <FWCore/Framework/interface/EventSetup.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <stdio.h>
#include <sstream>
#include <math.h>

#include "TCanvas.h"
#include "TFile.h"
#include "TProfile.h"

using namespace edm;
using namespace std;

DTCreateSummaryHistos::DTCreateSummaryHistos(const edm::ParameterSet& ps){

  edm::LogVerbatim ("histos") << "[DTCreateSummaryHistos]: Constructor";

  parameters = ps;

  // The root file which contain the histos
  string rootFileName = ps.getUntrackedParameter<string>("rootFileName", "DTEfficiencyTest.root");
  theFile = new TFile(rootFileName.c_str(), "READ");

   // The *.ps file which contains the summary histos
  PsFileName = ps.getUntrackedParameter<string>("PsFileName", "DTSummaryHistos");

  // Files to write in the *.ps file
  DataIntegrityHistos = ps.getUntrackedParameter<bool>("DataIntegrityHistos", false);
  DigiHistos = ps.getUntrackedParameter<bool>("DigiHistos", false);
  RecoHistos = ps.getUntrackedParameter<bool>("RecoHistos", false);
  ResoHistos = ps.getUntrackedParameter<bool>("ResoHistos", false);
  EfficiencyHistos = ps.getUntrackedParameter<bool>("EfficiencyHistos", false);
  TestPulsesHistos = ps.getUntrackedParameter<bool>("TestPulsesHistos", false);
  TriggerHistos = ps.getUntrackedParameter<bool>("TriggerHistos", false);

  // The DDU Id
  DDUId = ps.getUntrackedParameter<int>("DDUId");

  MainFolder = "DQMData/DT/";
}

DTCreateSummaryHistos::~DTCreateSummaryHistos(){

  edm::LogVerbatim ("histos") << "DTCreateSummaryHistos: analyzed " << nevents << " events";
  theFile->Close();

}


void DTCreateSummaryHistos::beginJob(){

  edm::LogVerbatim ("histos") << "[DTCreateSummaryHistos]: BeginJob";

  nevents = 0;

}

void DTCreateSummaryHistos::beginRun(const edm::Run& run, const edm::EventSetup& context){

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}

void DTCreateSummaryHistos::analyze(const edm::Event& e, const edm::EventSetup& context){
  
  nevents++;
  runNumber = e.id().run();

}


void DTCreateSummaryHistos::endJob(){

  edm::LogVerbatim ("histos") << "[DTCreateSummaryHistos] endjob called!";

  stringstream RunNum; RunNum << runNumber;
  string PsFileNameWithRun = PsFileName + "_" + RunNum.str() + ".ps";
  TPostScript psFile(PsFileNameWithRun.c_str(),111);
  psFile.Range(20,26);
  //psFile.NewPage();
  TCanvas c1("c1","",600,780);



  // DataIntegrity summary histos **************************************************************
  if(DataIntegrityHistos){
    c1.Clear();
    c1.Divide(2,2);    
    stringstream dduID; dduID << DDUId;

    string histo_name1 = MainFolder + "DataIntegrity/FED" + dduID.str() + "/FED" + dduID.str() + "_TTSValues";
    TH1F *histo1 = (TH1F*) theFile -> Get(histo_name1.c_str());
    c1.cd(1);
    if(histo1)
      histo1->Draw();
    
    string histo_name2 = MainFolder + "DataIntegrity/FED" + dduID.str() + "/FED" + dduID.str() + "_ROSStatus";
    TH1F *histo2 = (TH1F*) theFile -> Get(histo_name2.c_str());
    c1.cd(2);
    if(histo2)
      histo2->Draw();
    
    string histo_name3 = MainFolder + "DataIntegrity/FED" + dduID.str() + "/FED" + dduID.str() + "_ROSSummary";
    TH1F *histo3 = (TH1F*) theFile -> Get(histo_name3.c_str());
    c1.cd(3);
    if(histo3)
    histo3->Draw();
    
    string histo_name4 = MainFolder + "DataIntegrity/FED" + dduID.str() + "/ROS1/FED" + dduID.str() + "_ROS1_ROSError";
    TH1F *histo4 = (TH1F*) theFile -> Get(histo_name4.c_str());
    c1.cd(4);
    if(histo4)
      histo4->Draw();
    
    c1.Update();
    psFile.NewPage();
  }

  // Digi summary histos  ********************************************************************
  if(DigiHistos){
    // Time Box Histos
    c1.Clear();
    c1.Divide(3,4);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_digi_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_digi_end = muonGeom->chambers().end();
    for (; ch_digi_it != ch_digi_end; ++ch_digi_it) {
      DTChamberId ch = (*ch_digi_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_digi_it)->superLayers().begin(); 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_digi_it)->superLayers().end();
      // Loop over the SLs
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	stringstream superLayer; superLayer << sl.superlayer();
	
	string digiFolder = MainFolder + "DTDigiTask/Wheel" + wheel.str();
	string histo_name = digiFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/TimeBoxes/TimeBox_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str()+ "_SL" + superLayer.str();
	TH1F *histo = (TH1F *) theFile->Get(histo_name.c_str());
	if(histo){
	  int pad = (ch.station() - 1)*3 + sl.superlayer();
	  c1.cd(pad);
	  histo->Draw();
	}
      }
    }
    c1.Update();
    psFile.NewPage();
    
    // Occupancy in Time Histos
    c1.Clear();
    c1.Divide(4,3);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_digi2_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_digi2_end = muonGeom->chambers().end();
    for (; ch_digi2_it != ch_digi2_end; ++ch_digi2_it) {
      DTChamberId ch = (*ch_digi2_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      bool found = false;
      vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_digi2_it)->superLayers().begin(); 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_digi2_it)->superLayers().end();
      // Loop over the SLs
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 
	vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();
	stringstream superLayer; superLayer << sl.superlayer();
	// Loop over the Ls
	for(; l_it != l_end; ++l_it) {
	  DTLayerId layerId = (*l_it)->id();
	  stringstream layer; layer << layerId.layer();
	  
	  string digiFolder = MainFolder + "DTDigiTask/Wheel" + wheel.str();
	  string histo_name = digiFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/Occupancies/OccupancyInTimeHits_perL_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str()+ "_SL" + superLayer.str() + "_L" + layer.str();     
	  TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
	  if(histo){
	    found = true;
	    int pad = (sl.superlayer() - 1)*4 + layerId.layer();
	    c1.cd(pad);
	    histo->Draw();
	  }
	}
      }
      if(found){
	c1.Update();
	psFile.NewPage();
      }
    }
    
    // Occupancy Noise
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_digi3_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_digi3_end = muonGeom->chambers().end();
    for (; ch_digi3_it != ch_digi3_end; ++ch_digi3_it) {
      DTChamberId ch = (*ch_digi3_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      bool found = false;
      vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_digi3_it)->superLayers().begin(); 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_digi3_it)->superLayers().end();
      // Loop over the SLs
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 
	vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();
	stringstream superLayer; superLayer << sl.superlayer();
	// Loop over the Ls
	for(; l_it != l_end; ++l_it) {
	  DTLayerId layerId = (*l_it)->id();
	  stringstream layer; layer << layerId.layer();
	  
	  string digiFolder = MainFolder + "DTDigiTask/Wheel" + wheel.str();
	  string histo_name = digiFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/Occupancies/OccupancyNoise_perL_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str()+ "_SL" + superLayer.str() + "_L" + layer.str(); 
	  TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
	  if(histo){
	    found = true;
	    int pad = (sl.superlayer() - 1)*4 + layerId.layer();
	    c1.cd(pad);
	    histo->Draw();
	  }
	}
      }
      if(found) {
	c1.Update();
	psFile.NewPage();
      }
    }
    
    // Digi Per Event
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_digi4_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_digi4_end = muonGeom->chambers().end();
    for (; ch_digi4_it != ch_digi4_end; ++ch_digi4_it) {
      DTChamberId ch = (*ch_digi4_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      bool found = false;
      vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_digi4_it)->superLayers().begin(); 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_digi4_it)->superLayers().end();
      // Loop over the SLs
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 
	vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();
	stringstream superLayer; superLayer << sl.superlayer();
	// Loop over the Ls
	for(; l_it != l_end; ++l_it) {
	  DTLayerId layerId = (*l_it)->id();
	  stringstream layer; layer << layerId.layer();
	  
	  string digiFolder = MainFolder + "DTDigiTask/Wheel" + wheel.str();
	  string histo_name = digiFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/DigiPerEvent/DigiPerEvent_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() + "_SL" + superLayer.str() + "_L" + layer.str();
	  TProfile *histo = (TProfile*) theFile -> Get(histo_name.c_str());
	  if(histo){
	    found = true;
	    int pad = (sl.superlayer() - 1)*4 + layerId.layer();
	    c1.cd(pad);
	    histo->Draw();
	  }
	}
      }
      if(found){
	c1.Update();
	psFile.NewPage();
      }
    }
  }
  
  // Reconstruction summary histos  *********************************************************
  if(RecoHistos){
    // reco segment Histos - page1
    c1.Clear(); 
    c1.Divide(2,4);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_reco_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_reco_end = muonGeom->chambers().end();
    for (; ch_reco_it != ch_reco_end; ++ch_reco_it) {
      DTChamberId ch = (*ch_reco_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      
      string recoFolder = MainFolder + "DTSegmentAnalysisTask/Wheel" + wheel.str();
      string histo_name = recoFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/hN4DSeg_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();  
      TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
      if(histo){
	int pad = (ch.station() - 1)*2 + 1;
	c1.cd(pad);
	histo->Draw();
      }
      histo_name = recoFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/h4DSegmXvsYInCham_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();
      TProfile *histo2 = (TProfile*) theFile -> Get(histo_name.c_str());
      if(histo2){
	int pad = (ch.station() - 1)*2 + 2;
	c1.cd(pad);
	histo2->Draw();
      }
    }
    c1.Update();
    psFile.NewPage();
    
    // reco segment Histos - page2
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_reco2_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_reco2_end = muonGeom->chambers().end();
    for (; ch_reco2_it != ch_reco2_end; ++ch_reco2_it) {
      DTChamberId ch = (*ch_reco2_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      
      string recoFolder = MainFolder + "DTSegmentAnalysisTask/Wheel" + wheel.str();
      string histo_name = recoFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/h4DSegmPhiDirection_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();
      TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
      if(histo) {
	int pad = (ch.station() - 1)*2 + 1;
	c1.cd(pad);
	histo->Draw();
      }
      histo_name = recoFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/h4DSegmThetaDirection_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();
      TH1F *histo2 = (TH1F*) theFile -> Get(histo_name.c_str()); 
      if(histo2) {
	int pad = (ch.station() - 1)*2 + 2;
	c1.cd(pad);
	histo2->Draw();
      }
    }      
    c1.Update();
    psFile.NewPage();
  }
    
  // Resolution summary histos  *******************************************************************
  if(ResoHistos){
    // Residuals histos
    c1.Clear();
    c1.Divide(3,4);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_reso_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_reso_end = muonGeom->chambers().end();
    for (; ch_reso_it != ch_reso_end; ++ch_reso_it) {
      DTChamberId ch = (*ch_reso_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_reso_it)->superLayers().begin(); 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_reso_it)->superLayers().end();
      // Loop over the SLs
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	stringstream superLayer; superLayer << sl.superlayer();
	
	string resoFolder = MainFolder + "DTResolutionAnalysisTask/Wheel" + wheel.str();
	string histo_name = resoFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/hResDist_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() + "_SL" + superLayer.str();  
	TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
	if(histo){
	  int pad = (ch.station() - 1)*3 + sl.superlayer();
	  c1.cd(pad);
	  histo->Draw();
	}
      }
    }
    c1.Update();
    psFile.NewPage();
    
    // Residuals as a function of the position Histos
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_reso2_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_reso2_end = muonGeom->chambers().end();
    for (; ch_reso2_it != ch_reso2_end; ++ch_reso2_it) {
      DTChamberId ch = (*ch_reso2_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_reso2_it)->superLayers().begin(); 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_reso2_it)->superLayers().end();
      // Loop over the SLs
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	stringstream superLayer; superLayer << sl.superlayer();
	
	string resoFolder = MainFolder + "DTResolutionAnalysisTask/Wheel" + wheel.str();
	string histo_name = resoFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/hResDistVsDist_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() + "_SL" + superLayer.str();
	TProfile *histo = (TProfile*) theFile -> Get(histo_name.c_str());
	if(histo){
	  int pad = (ch.station() - 1)*3 + sl.superlayer();
	  c1.cd(pad);
	  histo->Draw();
	}
      }
    }
    c1.Update();
    psFile.NewPage();
  }
  
  // Efficiency summary histos  ******************************************************************
  if(EfficiencyHistos){
    // Cell efficiency
    c1.Clear();
    c1.Divide(4,3);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_eff_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_eff_end = muonGeom->chambers().end();
    for (; ch_eff_it != ch_eff_end; ++ch_eff_it) {
      DTChamberId ch = (*ch_eff_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      bool found = false;
      vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_eff_it)->superLayers().begin(); 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_eff_it)->superLayers().end();
      // Loop over the SLs
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 
	vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();
	stringstream superLayer; superLayer << sl.superlayer();
	// Loop over the Ls
	for(; l_it != l_end; ++l_it) {
	  DTLayerId layerId = (*l_it)->id();
	  stringstream layer; layer << layerId.layer();
	  
	  string efficiencyFolder = MainFolder + "Tests/DTEfficiency/Wheel" + wheel.str();
	  string histo_name = efficiencyFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/UnassEfficiency_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() + "_SL" + superLayer.str() + "_L" + layer.str();
	  TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
	  if(histo){
	    found = true;
	    int pad = (sl.superlayer() - 1)*4 + layerId.layer();
	    c1.cd(pad);
	    histo->Draw();
	  }
	}
      }
      if(found){
	c1.Update();
	psFile.NewPage();
      }
    }
    
    // Chamber X efficiency
    c1.Clear();
    c1.Divide(2,2);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_eff2_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_eff2_end = muonGeom->chambers().end();
    for (; ch_eff2_it != ch_eff2_end; ++ch_eff2_it) {
      DTChamberId ch = (*ch_eff2_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      
      string efficiencyFolder = MainFolder + "Tests/DTChamberEfficiency/Wheel" + wheel.str();
      string histo_name = efficiencyFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/xEfficiency_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();
      TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
      if(histo){
	c1.cd(ch.station());
	histo->Draw();
      }
    }
    c1.Update();
    psFile.NewPage();
    
    // Chamber Y efficiency
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_eff3_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_eff3_end = muonGeom->chambers().end();
    for (; ch_eff3_it != ch_eff3_end; ++ch_eff3_it) {
      DTChamberId ch = (*ch_eff3_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      
      string efficiencyFolder = MainFolder + "Tests/DTChamberEfficiency/Wheel" + wheel.str();
      string histo_name = efficiencyFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/yEfficiency_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();
      TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
      if(histo){
	c1.cd(ch.station());
	histo->Draw();
      }
    }
    c1.Update();
    psFile.NewPage();
  }

  // Test Pulses Summary Histos  **************************************************************
  if(TestPulsesHistos){
    c1.Clear();
    c1.Divide(4,3);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_TP_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_TP_end = muonGeom->chambers().end();
    for (; ch_TP_it != ch_TP_end; ++ch_TP_it) {
      DTChamberId ch = (*ch_TP_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      bool found = false;
      vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_TP_it)->superLayers().begin(); 
      vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_TP_it)->superLayers().end();
      // Loop over the SLs
      for(; sl_it != sl_end; ++sl_it) {
	DTSuperLayerId sl = (*sl_it)->id();
	vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 
	vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();
	stringstream superLayer; superLayer << sl.superlayer();
	// Loop over the Ls
	for(; l_it != l_end; ++l_it) {
	  DTLayerId layerId = (*l_it)->id();
	  stringstream layer; layer << layerId.layer();
	  
	  string testPulsesFolder = MainFolder + "DTTestPulsesTask/Wheel" + wheel.str();
	  string histo_name = testPulsesFolder + "/Station" + station.str() + "/Sector" + sector.str() + "/SuperLayer" + superLayer.str() +  "/TPProfile/TestPulses2D_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() + "_SL" + superLayer.str() + "_L" + layer.str();
	  TProfile *histo = (TProfile*) theFile -> Get(histo_name.c_str());
	  if(histo){
	    found = true;
	    int pad = (sl.superlayer() - 1)*4 + layerId.layer();
	    c1.cd(pad);
	    histo->Draw();
	  }
      }
      }
      if(found){
	c1.Update();
	psFile.NewPage();
      }
    }
  }

  // Trigger Summary Histos ************************************************************************
  if(TriggerHistos){
    c1.Clear();
    c1.Divide(2,2);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_trigger_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_trigger_end = muonGeom->chambers().end();
    for (; ch_trigger_it != ch_trigger_end; ++ch_trigger_it) {
      DTChamberId ch = (*ch_trigger_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      
      string triggerFolder = MainFolder + "DTLocalTriggerTask/Wheel" + wheel.str();
      string histo_name = triggerFolder + "/Sector" + sector.str() + "/Station" + station.str() + "/LocalTriggerPhi/DDU_BXvsQual_W" +  wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
      TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
      if(histo){
	c1.cd(ch.station());
	histo->Draw();
      }
    }
    c1.Update();
    psFile.NewPage();
    
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_trigger2_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_trigger2_end = muonGeom->chambers().end();
    for (; ch_trigger2_it != ch_trigger2_end; ++ch_trigger2_it) {
      DTChamberId ch = (*ch_trigger2_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      
      string triggerFolder = MainFolder + "DTLocalTriggerTask/Wheel" + wheel.str();
      string histo_name = triggerFolder + "/Sector" + sector.str() + "/Station" + station.str() + "/LocalTriggerTheta/DDU_ThetaBXvsQual_W" +  wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
      TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
      if(histo){
	c1.cd(ch.station());
	histo->Draw();
      }
    }
    c1.Update();
    psFile.NewPage();
    
    c1.Clear();
    c1.Divide(1,2);
    int counter1=0,counter2=0;
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_trigger3_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_trigger3_end = muonGeom->chambers().end();
    for (; ch_trigger3_it != ch_trigger3_end; ++ch_trigger3_it) {
      DTChamberId ch = (*ch_trigger3_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream sector; sector << ch.sector();
      
      string triggerFolder = MainFolder + "Tests/DTLocalTrigger/Wheel" + wheel.str();
      string histo_Name = triggerFolder + "/Sector" + sector.str() + "/LocalTriggerPhi/CorrFraction_Phi_W" +  wheel.str() + "_Sec" + sector.str();
      TH1F *Histo1 = (TH1F*) theFile -> Get(histo_Name.c_str());
      if(Histo1 && counter1==0){
	counter1++;
	c1.cd(1);
	Histo1->Draw();
      }
      histo_Name = triggerFolder + "/Sector" + sector.str() + "/LocalTriggerTheta/HFraction_Theta_W" +  wheel.str() + "_Sec" + sector.str();
      TH1F *Histo2 = (TH1F*) theFile -> Get(histo_Name.c_str());
      if(Histo2 && counter2==0){
	counter2++;
	c1.cd(2);
	Histo2->Draw();
      }
    }
    c1.Update();
    psFile.NewPage();
    
    c1.Clear();
    c1.Divide(2,2);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_trigger4_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_trigger4_end = muonGeom->chambers().end();
    for (; ch_trigger4_it != ch_trigger4_end; ++ch_trigger4_it) {
      DTChamberId ch = (*ch_trigger4_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      
      string triggerFolder = MainFolder + "Tests/DTLocalTrigger/Wheel" + wheel.str();
      string histo_name = triggerFolder + "/Sector" + sector.str() + "/Station" + station.str() + "/TrigEffPos_Phi_W" +  wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
      TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
      string histo_nameHHHL = triggerFolder + "/Sector" + sector.str() + "/Station" + station.str() + "/TrigEffPosHHHL_Phi_W" +  wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
      TH1F *histoHHHL = (TH1F*) theFile -> Get(histo_nameHHHL.c_str());
      if(histo){
	c1.cd(ch.station());
	histo->GetYaxis()->SetRangeUser(0,1.1);
	histo->Draw();
	if(histoHHHL){
		histoHHHL->Draw("same");
	}
      }
    }
    c1.Update();
    psFile.NewPage();

    c1.Clear();
    c1.Divide(2,2);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_trigger5_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_trigger5_end = muonGeom->chambers().end();
    for (; ch_trigger5_it != ch_trigger5_end; ++ch_trigger5_it) {
      DTChamberId ch = (*ch_trigger5_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      
      string triggerFolder = MainFolder + "Tests/DTLocalTrigger/Wheel" + wheel.str();
      string histo_name = triggerFolder + "/Sector" + sector.str() + "/Station" + station.str() + "/TrigEffAngle_Phi_W" +  wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
      TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
      if(histo){
	c1.cd(ch.station());
	histo->GetYaxis()->SetRangeUser(0,1.1);
	histo->Draw();
      }
    }
    c1.Update();
    psFile.NewPage();
    
    c1.Clear();
    c1.Divide(2,2);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_trigger6_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_trigger6_end = muonGeom->chambers().end();
    for (; ch_trigger6_it != ch_trigger6_end; ++ch_trigger6_it) {
      DTChamberId ch = (*ch_trigger6_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      
      string triggerFolder = MainFolder + "Tests/DTLocalTrigger/Wheel" + wheel.str();
      string histo_name = triggerFolder + "/Sector" + sector.str() + "/Station" + station.str() + "/TrigEffPos_Theta_W" +  wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
      TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
      if(histo){
	c1.cd(ch.station());
	histo->GetYaxis()->SetRangeUser(0,1.1);
	histo->Draw();
      }
    }
    c1.Update();
    psFile.NewPage();

    c1.Clear();
    c1.Divide(2,2);
    // Loop over all the chambers
    vector<DTChamber*>::const_iterator ch_trigger7_it = muonGeom->chambers().begin();
    vector<DTChamber*>::const_iterator ch_trigger7_end = muonGeom->chambers().end();
    for (; ch_trigger7_it != ch_trigger7_end; ++ch_trigger7_it) {
      DTChamberId ch = (*ch_trigger7_it)->id();
      stringstream wheel; wheel << ch.wheel();
      stringstream station; station << ch.station();
      stringstream sector; sector << ch.sector();
      
      string triggerFolder = MainFolder + "Tests/DTLocalTrigger/Wheel" + wheel.str();
      string histo_name = triggerFolder + "/Sector" + sector.str() + "/Station" + station.str() + "/TrigEffAngle_Theta_W" +  wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
      TH1F *histo = (TH1F*) theFile -> Get(histo_name.c_str());
      if(histo){
	histo->GetYaxis()->SetRangeUser(0,1.1);
	c1.cd(ch.station());
	histo->Draw();
      }
    }
    c1.Update();
  }
  psFile.Close();
}
