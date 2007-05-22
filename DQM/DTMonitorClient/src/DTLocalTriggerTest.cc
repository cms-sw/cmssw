
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/05/18 08:07:47 $
 *  $Revision: 1.1 $
 *  \author C. Battilana S. Marcellini - INFN Bologna
 */


#include <DQM/DTMonitorClient/src/DTLocalTriggerTest.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DQMServices/Core/interface/MonitorElementBaseT.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Geometry
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <iostream>
#include <sstream>

using namespace edm;
using namespace std;

DTLocalTriggerTest::DTLocalTriggerTest(const edm::ParameterSet& ps){

  sourceFolder = ps.getUntrackedParameter<string>("sourceFolder", ""); 
  edm::LogVerbatim ("localTrigger") << "[DTLocalTriggerTest]: Constructor";

  parameters = ps;

  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);

}

DTLocalTriggerTest::~DTLocalTriggerTest(){

  edm::LogVerbatim ("localTrigger") << "[DTLocalTriggerTest]: analyzed " << nevents << " events";

}

void DTLocalTriggerTest::endJob(){

  edm::LogVerbatim ("localTrigger") << "[DTLocalTriggerTest] endjob called!";

  dbe->rmdir("DT/Tests/DTLocalTrigger");

}

void DTLocalTriggerTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("localTrigger") << "[DTLocalTriggerTest]: BeginJob";

  nevents = 0;

}

void DTLocalTriggerTest::bookWheelHistos(int wheel, string folder) {
  
  stringstream  wh;
  wh << wheel;
  dbe->setCurrentFolder("DT/Tests/DTLocalTrigger/Wheel"+ wh.str()+"/"+folder);

  if (folder =="LocalTriggerPhi"){
    string htype = "WorkingTrigUnits_Phi";
    string hname = htype + "_Wh" + wh.str();
    phiME[wheel][htype] =  dbe->book2D(hname.c_str(),hname.c_str(),12,1,13,4,1,5);
    htype = "CorrectBX_Phi";
    hname = htype + "_Wh" + wh.str();
    phiME[wheel][htype] = dbe->book1D(hname.c_str(),hname.c_str(),48,1,49);
    htype = "CorrFraction_Phi";
    hname = htype + "_Wh" + wh.str();
    phiME[wheel][htype] = dbe->book1D(hname.c_str(),hname.c_str(),48,1,49);
    htype = "2ndFraction_Phi";
    hname = htype + "_Wh" + wh.str();
    phiME[wheel][htype] = dbe->book1D(hname.c_str(),hname.c_str(),48,1,49);
  }
  
  if (folder =="LocalTriggerTheta"){
    string htype = "WorkingTrigUnits_Theta";
    string hname = htype + "_Wh" + wh.str();
    thetaME[wheel][htype] =  dbe->book2D(hname.c_str(),hname.c_str(),12,1,13,4,1,5);
    htype = "CorrectBX_Theta";
    hname = htype + "_Wh" + wh.str();
    thetaME[wheel][htype] = dbe->book1D(hname.c_str(),hname.c_str(),36,1,37);
    htype = "HFraction_Theta";
    hname = htype + "_Wh" + wh.str();
    thetaME[wheel][htype] = dbe->book1D(hname.c_str(),hname.c_str(),36,1,37);
  }

  if (folder =="TriggerAndSeg"){
    string htype = "Efficiency_Phi";
    string hname = htype + "_Wh" + wh.str();
    segME[wheel]= dbe->book1D(hname.c_str(),hname.c_str(),48,1,49);
  }
  
}



void DTLocalTriggerTest::analyze(const edm::Event& e, const edm::EventSetup& context){
  
  nevents++;
  if (nevents%1000 == 0) 
    edm::LogVerbatim ("localTrigger") <<"[DTLocalTriggerTest]: "<<nevents<<" updates";
  
  // Loop over the chambers
  for (int stat=1; stat<=4; ++stat){
    for (int wh=-2; wh<=2; ++wh){
      for (int sect=1; sect<=12; ++sect){
	DTChamberId chID(wh,stat,sect);
	int pos_ph = (sect-1)*4+stat; 
	int pos_th = (sect-1)*3+stat; 
	
	// Get the ME produced by EfficiencyTask Source
	MonitorElement * DDU_BXvsQual_ME = dbe->get(getMEName("DDU_BXvsQual","LocalTriggerPhi", chID));	
	MonitorElement * DDU_Flag1stvsBX_ME = dbe->get(getMEName("DDU_Flag1stvsBX","LocalTriggerPhi", chID));
	
	// ME -> TH1F
	if(DDU_BXvsQual_ME && DDU_Flag1stvsBX_ME) {
	  
	  MonitorElementT<TNamed>* DDU_BXvsQual    = dynamic_cast<MonitorElementT<TNamed>*>(DDU_BXvsQual_ME);
	  MonitorElementT<TNamed>* DDU_Flag1stvsBX = dynamic_cast<MonitorElementT<TNamed>*>(DDU_Flag1stvsBX_ME);
	  
	  if (DDU_BXvsQual && DDU_Flag1stvsBX ) {
	  
	    TH2F * DDU_BXvsQual_histo    = dynamic_cast<TH2F*> (DDU_BXvsQual->operator->());
	    TH2F * DDU_Flag1stvsBX_histo = dynamic_cast<TH2F*> (DDU_Flag1stvsBX->operator->());
	    
	    if (DDU_BXvsQual_histo && DDU_Flag1stvsBX_histo) {
	      
	      TH1D* proj_BXHH    = DDU_BXvsQual_histo->ProjectionY("",7,7,"");
	      TH1D* proj_Flag1st = DDU_Flag1stvsBX_histo->ProjectionY();
	      TH1D* proj_Qual    = DDU_BXvsQual_histo->ProjectionX();
	  
	      int BXOK_bin = proj_BXHH->GetMaximumBin();
	      double BX_OK =  proj_BXHH->GetBinCenter(BXOK_bin);
	      double Flag2nd_trigs = proj_Flag1st->GetBinContent(2);
	      double trigs = proj_Flag1st->GetEntries(); 
	      double Corr_trigs = 0;
	      for (int i=5;i<=7;++i)
		Corr_trigs+=proj_Qual->GetBinContent(i);
	      
	      // Fill client histos
	      if( phiME.find(wh) == phiME.end() ){
		bookWheelHistos(wh,"LocalTriggerPhi"); 
	      }
	      std::map<std::string,MonitorElement*> innerME = phiME.find(wh)->second;
	      innerME.find("WorkingTrigUnits_Phi")->second->setBinContent(sect,stat,trigs);
	      innerME.find("CorrectBX_Phi")->second->setBinContent(pos_ph,BX_OK);
	      innerME.find("CorrFraction_Phi")->second->setBinContent(pos_ph,Corr_trigs/trigs);
	      innerME.find("2ndFraction_Phi")->second->setBinContent(pos_ph,Flag2nd_trigs/trigs);
	    
	    }
	  }
	}  
	
	// Get the ME produced by EfficiencyTask Source
	MonitorElement * DDU_BXvsThQual_ME = dbe->get(getMEName("DDU_Theta_BXvsQual","LocalTriggerTheta", chID));	
	
	// ME -> TH1F
	if(DDU_BXvsThQual_ME) {
	  MonitorElementT<TNamed>* DDU_BXvsThQual    = dynamic_cast<MonitorElementT<TNamed>*>(DDU_BXvsThQual_ME);
	  
	  if (DDU_BXvsThQual) {
	    TH2F * DDU_BXvsThQual_histo = dynamic_cast<TH2F*> (DDU_BXvsThQual->operator->());
	    
	    if (DDU_BXvsThQual_histo) {
	      
	      TH1D* proj_BXH    = DDU_BXvsThQual_histo->ProjectionY("",4,4,""); //guarda cosa metterci
	      TH1D* proj_Qual    = DDU_BXvsThQual_histo->ProjectionX();
	      
	      int    BXOK_bin = proj_BXH->GetMaximumBin();
	      double BX_OK    = proj_BXH->GetBinCenter(BXOK_bin);
	      double trigs    = proj_Qual->GetEntries(); 
	      double H_trigs  = proj_Qual->GetBinContent(4);
	      
	      // Fill client histos
	      if( thetaME.find(wh) == thetaME.end() ){
		bookWheelHistos(wh,"LocalTriggerTheta"); 
	      }
	      std::map<std::string,MonitorElement*> innerME = thetaME.find(wh)->second;
	      innerME.find("WorkingTrigUnits_Theta")->second->setBinContent(sect,stat,trigs);
	      innerME.find("CorrectBX_Theta")->second->setBinContent(pos_th,BX_OK);
	      innerME.find("HFraction_Theta")->second->setBinContent(pos_th,H_trigs/trigs);
	    
	    }
	  }
	}
	// Get the ME produced by EfficiencyTask Source
	MonitorElement * Track_pos_ME = dbe->get(getMEName("Track_pos","LocalTriggerPhi", chID));	
	MonitorElement * DDU_Track_pos_andtrig_ME = dbe->get(getMEName("DDU_Track_pos_andtrig","LocalTriggerPhi", chID));	
	
	// ME -> TH1F
	if(Track_pos_ME && DDU_Track_pos_andtrig_ME) {
	  MonitorElementT<TNamed>* Track_pos                = dynamic_cast<MonitorElementT<TNamed>*>(Track_pos_ME);
	  MonitorElementT<TNamed>* DDU_Track_pos_andtrig    = dynamic_cast<MonitorElementT<TNamed>*>(DDU_Track_pos_andtrig_ME);
	  
	  if (DDU_Track_pos_andtrig && Track_pos) {
	    TH1F * Track_pos_histo             = dynamic_cast<TH1F*> (Track_pos->operator->());
	    TH1F * DDU_Track_pos_andtrig_histo = dynamic_cast<TH1F*> (DDU_Track_pos_andtrig->operator->());
	    
	    if (Track_pos_histo && DDU_Track_pos_andtrig_histo) {
	      
	      // Fill client histos
	      if( segME.find(wh) == segME.end() ){
		bookWheelHistos(wh,"TriggerAndSeg"); 
	      }
	      segME.find(wh)->second->setBinContent(pos_ph,double(DDU_Track_pos_andtrig_histo->GetEntries())/Track_pos_histo->GetEntries());
	    
	    }
	  }
	}
      }
    }
  }
  
  if (nevents%parameters.getUntrackedParameter<int>("resultsSavingRate",10) == 0){
    if ( parameters.getUntrackedParameter<bool>("writeHisto", true) ) 
      dbe->save(parameters.getUntrackedParameter<string>("outputFile", "DTLocalTriggerTest.root"));
  }
}

    
string DTLocalTriggerTest::getMEName(string histoTag, string subfolder, const DTChamberId & chambid) {

  stringstream wheel; wheel << chambid.wheel();
  stringstream station; station << chambid.station();
  stringstream sector; sector << chambid.sector();

  string folderName = 
    "DT/DTLocalTriggerTask/Wheel" +  wheel.str() +
    "/Sector" + sector.str() +
    "/Station" + station.str() + "/" +  subfolder + "/";
  
  string histoname = sourceFolder + folderName + histoTag  
    + "_W" + wheel.str()  
    + "_Sec" + sector.str()
    + "_St" + station.str();
  
  return histoname;
  
}
