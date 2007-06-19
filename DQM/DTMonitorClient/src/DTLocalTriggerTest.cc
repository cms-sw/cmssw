
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/05/22 07:40:04 $
 *  $Revision: 1.2 $
 *  \author C. Battilana S. Marcellini - INFN Bologna
 */


// This class header
#include "DQM/DTMonitorClient/src/DTLocalTriggerTest.h"

// Framework headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

//C++ headers
#include <iostream>
#include <sstream>

using namespace edm;
using namespace std;


DTLocalTriggerTest::DTLocalTriggerTest(const edm::ParameterSet& ps){

  edm::LogVerbatim ("localTrigger") << "[DTLocalTriggerTest]: Constructor";

  sourceFolder = ps.getUntrackedParameter<string>("sourceFolder", ""); 
  hwSource = ps.getUntrackedParameter<bool>("dataFromDDU", false) ? "DDU" : "DCC" ; 
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
  context.get<MuonGeometryRecord>().get(muonGeom);
  
}

void DTLocalTriggerTest::bookChambHistos(DTChamberId chambId, string htype) {
  
  stringstream wheel; wheel << chambId.wheel();
  stringstream station; station << chambId.station();	
  stringstream sector; sector << chambId.sector();

  string HistoName = htype + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  dbe->setCurrentFolder("DT/Tests/DTLocalTrigger/Wheel" + wheel.str() +
			"/Sector" + sector.str() +
			"/Station" + station.str());
  
  uint32_t indexChId = chambId.rawId();
  if (htype.find("Angle") != string::npos){
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),50,-2.,2.);
    return;
  }
  if (htype.find("TrigEffPos_Phi") == 0 ){
    pair<float,float> range = phiRange(chambId);
    int nbins = int((range.second - range.first)/15);
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),nbins,range.first,range.second);
    return;
  }
  if (htype.find("TrigEffPos_Theta") == 0){
    chambME[indexChId][htype] = dbe->book1D(HistoName.c_str(),HistoName.c_str(),40,-117.5,117.5);
    return;
  }

}

void DTLocalTriggerTest::bookSectorHistos(int wheel,int sector,string folder, string htype) {
  
  stringstream wh; wh << wheel;
  stringstream sc; sc << sector;
  int sectorid = (wheel+3) + (sector-1)*5;
  dbe->setCurrentFolder("DT/Tests/DTLocalTrigger/Wheel"+ wh.str()+"/Sector"+ sc.str()+"/"+folder);

  if (htype.find("Phi") != string::npos){    
    string hname = htype + "_W" + wh.str()+"_Sec" +sc.str();
    MonitorElement* me = dbe->book1D(hname.c_str(),hname.c_str(),4,1,5);
    //setLabelPh(me);
    secME[sectorid][htype] = me;
    return;
  }
  
  if (htype.find("Theta") != string::npos){
    string hname = htype + "_W" + wh.str()+ "_Sec"+sc.str();
    MonitorElement* me =dbe->book1D(hname.c_str(),hname.c_str(),3,1,4);
    //setLabelTh(me);
    secME[sectorid][htype] = me;
    return;
  }
  
}

pair<float,float> DTLocalTriggerTest::phiRange(const DTChamberId& id){

  float min,max;
  int station = id.station();
  int sector  = id.sector(); 
  int wheel   = id.wheel();
  
  const DTLayer  *layer = muonGeom->layer(DTLayerId(id,1,1));
  DTTopology topo = layer->specificTopology();
  min = topo.wirePosition(topo.firstChannel());
  max = topo.wirePosition(topo.lastChannel());

  if (station == 4){
    
    const DTLayer *layer2;
    float lposx;
    
    if (sector == 4){
      layer2  = muonGeom->layer(DTLayerId(wheel,station,13,1,1));
      lposx = layer->toLocal(layer2->position()).x();
    }
    else if (sector == 10){
      layer2 = muonGeom->layer(DTLayerId(wheel,station,14,1,1));
      lposx = layer->toLocal(layer2->position()).x();
    }
    else
      return make_pair(min,max);
    
    DTTopology topo2 = layer2->specificTopology();

    if (lposx>0)
      max = lposx+topo2.wirePosition(topo2.lastChannel());
    else 
      min = lposx+topo2.wirePosition(topo2.firstChannel());
  }
  
  return make_pair(min,max);

}

void DTLocalTriggerTest::analyze(const edm::Event& e, const edm::EventSetup& context){
  
  nevents++;
  if (nevents%1000 == 0)
    edm::LogVerbatim ("localTrigger") <<"[DTLocalTriggerTest]: "<<nevents<<" updates";
  
  // Loop over the TriggerUnits
  for (int stat=1; stat<=4; ++stat){
    for (int wh=-2; wh<=2; ++wh){
      for (int sect=1; sect<=12; ++sect){
	DTChamberId chID(wh,stat,sect);
	int sector_id = (wh+3)+(sect-1)*5;
	
	// Get the ME produced by DTLocalTriggeTask Source (Phi ones)
	MonitorElement * DDU_BXvsQual_ME = dbe->get(getMEName("BXvsQual","LocalTriggerPhi", chID));
	MonitorElement * DDU_Flag1stvsBX_ME = dbe->get(getMEName("Flag1stvsBX","LocalTriggerPhi", chID));
	
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
	      double BX_OK =  DDU_BXvsQual_histo->GetYaxis()->GetBinCenter(BXOK_bin);
	      double Flag2nd_trigs = proj_Flag1st->GetBinContent(2);
	      double trigs = proj_Flag1st->GetEntries(); 
	      double Corr_trigs = 0;
	      for (int i=5;i<=7;++i)
		Corr_trigs+=proj_Qual->GetBinContent(i);
	      
	      // Fill client histos
	      if( secME[sector_id].find("CorrectBX_Phi") == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"LocalTriggerPhi","CorrectBX_Phi");
		bookSectorHistos(wh,sect,"LocalTriggerPhi","CorrFraction_Phi");
		bookSectorHistos(wh,sect,"LocalTriggerPhi","2ndFraction_Phi");
	      }
	      std::map<std::string,MonitorElement*> innerME = secME[sector_id];
	      innerME.find("CorrectBX_Phi")->second->setBinContent(stat,BX_OK);
	      innerME.find("CorrFraction_Phi")->second->setBinContent(stat,Corr_trigs/trigs);
	      innerME.find("2ndFraction_Phi")->second->setBinContent(stat,Flag2nd_trigs/trigs);
	    
	    }
	  }
	}  
	
	// Get the ME produced by DTLocalTriggerTask Source (Theta ones)
	MonitorElement * DDU_BXvsThQual_ME = dbe->get(getMEName("ThetaBXvsQual","LocalTriggerTheta", chID));	
	if(DDU_BXvsThQual_ME) {
	  
	  MonitorElementT<TNamed>* DDU_BXvsThQual    = dynamic_cast<MonitorElementT<TNamed>*>(DDU_BXvsThQual_ME);
	  if (DDU_BXvsThQual) {
	    
	    TH2F * DDU_BXvsThQual_histo = dynamic_cast<TH2F*> (DDU_BXvsThQual->operator->());
	    if (DDU_BXvsThQual_histo) {
	      
	      TH1D* proj_BXH    = DDU_BXvsThQual_histo->ProjectionY("",4,4,"");
	      TH1D* proj_Qual    = DDU_BXvsThQual_histo->ProjectionX();
	      int    BXOK_bin = proj_BXH->GetMaximumBin();
	      double BX_OK    = DDU_BXvsThQual_histo->GetYaxis()->GetBinCenter(BXOK_bin);
	      double trigs    = proj_Qual->GetEntries(); 
	      double H_trigs  = proj_Qual->GetBinContent(4);
	      
	      // Fill client histos
	      if( secME[sector_id].find("HFraction_Theta") == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"LocalTriggerTheta","CorrectBX_Theta");
		bookSectorHistos(wh,sect,"LocalTriggerTheta","HFraction_Theta");
	      }
	      std::map<std::string,MonitorElement*> innerME = secME.find(sector_id)->second;
	      innerME.find("CorrectBX_Theta")->second->setBinContent(stat,BX_OK);
	      innerME.find("HFraction_Theta")->second->setBinContent(stat,H_trigs/trigs);
	    
	    }
	  }
	}

	// Get the ME produced by DTLocalTriggerTask Source (Phi+Segments)
	MonitorElement * Track_pos_ME = dbe->get(getMEName("TrackPos","Segment", chID));	
	MonitorElement * Track_pos_andtrig_ME = dbe->get(getMEName("TrackPosandTrig","Segment", chID));	
	MonitorElement * Track_angle_ME = dbe->get(getMEName("TrackAngle","Segment", chID));	
	MonitorElement * Track_angle_andtrig_ME = dbe->get(getMEName("TrackAngleandTrig","Segment", chID));		

	// ME -> TH1F
	if(Track_pos_ME && Track_pos_andtrig_ME && Track_angle_ME && Track_angle_andtrig_ME) {
	  MonitorElementT<TNamed>* Track_pos           = dynamic_cast<MonitorElementT<TNamed>*>(Track_pos_ME);
	  MonitorElementT<TNamed>* Track_pos_andtrig   = dynamic_cast<MonitorElementT<TNamed>*>(Track_pos_andtrig_ME);
	  MonitorElementT<TNamed>* Track_angle         = dynamic_cast<MonitorElementT<TNamed>*>(Track_angle_ME);
	  MonitorElementT<TNamed>* Track_angle_andtrig = dynamic_cast<MonitorElementT<TNamed>*>(Track_angle_andtrig_ME);
	  
	  if (Track_pos_andtrig && Track_pos && Track_angle_andtrig && Track_angle) {
	    TH1F * Track_pos_histo             = dynamic_cast<TH1F*> (Track_pos->operator->());
	    TH1F * Track_pos_andtrig_histo     = dynamic_cast<TH1F*> (Track_pos_andtrig->operator->());
	    TH1F * Track_angle_histo           = dynamic_cast<TH1F*> (Track_angle->operator->());
	    TH1F * Track_angle_andtrig_histo   = dynamic_cast<TH1F*> (Track_angle_andtrig->operator->());
	    
	    if (Track_pos_histo && Track_pos_andtrig_histo && Track_angle_histo && Track_angle_andtrig_histo) {
	      
	      // Fill client histos
	      if( secME[sector_id].find("TrigEff_Phi") == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"TriggerAndSeg","TrigEff_Phi");  
	      }
	      std::map<std::string,MonitorElement*> innerME = secME[sector_id];
	      MonitorElement* globaleff = innerME.find("TrigEff_Phi")->second;
	      float bineff = float(Track_pos_andtrig_histo->GetEntries())/Track_pos_histo->GetEntries();
	      float binerr = sqrt(bineff*(1-bineff)/Track_pos_histo->GetEntries());
	      globaleff->setBinContent(stat,bineff);
	      globaleff->setBinError(stat,binerr);
	      DTChamberId dtChId(wh,stat,sect);
	      uint32_t indexCh = dtChId.rawId();
	      if( chambME[indexCh].find("TrigEffAngle_Phi") == chambME[indexCh].end()){
		bookChambHistos(dtChId,"TrigEffPos_Phi");
		bookChambHistos(dtChId,"TrigEffAngle_Phi");
	      }
	      innerME = chambME[indexCh];
	      makeEfficiencyME(Track_pos_andtrig_histo,Track_pos_histo,innerME.find("TrigEffPos_Phi")->second);
	      makeEfficiencyME(Track_angle_andtrig_histo,Track_angle_histo,innerME.find("TrigEffAngle_Phi")->second);

	    }
	  }
	}
	
	// Get the ME produced by DTLocalTriggerTask Source (Theta+Segments)
	MonitorElement * Track_thpos_ME = dbe->get(getMEName("TrackThetaPos","Segment", chID));	
	MonitorElement * Track_thpos_andtrig_ME = dbe->get(getMEName("TrackThetaPosandTrig","Segment", chID));	
	MonitorElement * Track_thangle_ME = dbe->get(getMEName("TrackThetaAngle","Segment", chID));	
	MonitorElement * Track_thangle_andtrig_ME = dbe->get(getMEName("TrackThetaAngleandTrig","Segment", chID));		

	// ME -> TH1F
	if(Track_thpos_ME && Track_thpos_andtrig_ME && Track_thangle_ME && Track_thangle_andtrig_ME) {
	  MonitorElementT<TNamed>* Track_thpos           = dynamic_cast<MonitorElementT<TNamed>*>(Track_thpos_ME);
	  MonitorElementT<TNamed>* Track_thpos_andtrig   = dynamic_cast<MonitorElementT<TNamed>*>(Track_thpos_andtrig_ME);
	  MonitorElementT<TNamed>* Track_thangle         = dynamic_cast<MonitorElementT<TNamed>*>(Track_thangle_ME);
	  MonitorElementT<TNamed>* Track_thangle_andtrig = dynamic_cast<MonitorElementT<TNamed>*>(Track_thangle_andtrig_ME);
	  
	  if (Track_thpos_andtrig && Track_thpos && Track_thangle_andtrig && Track_thangle) {
	    TH1F * Track_thpos_histo             = dynamic_cast<TH1F*> (Track_thpos->operator->());
	    TH1F * Track_thpos_andtrig_histo     = dynamic_cast<TH1F*> (Track_thpos_andtrig->operator->());
	    TH1F * Track_thangle_histo           = dynamic_cast<TH1F*> (Track_thangle->operator->());
	    TH1F * Track_thangle_andtrig_histo   = dynamic_cast<TH1F*> (Track_thangle_andtrig->operator->());
	    
	    if (Track_thpos_histo && Track_thpos_andtrig_histo && Track_thangle_histo && Track_thangle_andtrig_histo) {
	      
	      // Fill client histos
	      if( secME[sector_id].find("TrigEff_Theta") == secME[sector_id].end() ){
		bookSectorHistos(wh,sect,"TriggerAndSeg","TrigEff_Theta");  
	      }
	      std::map<std::string,MonitorElement*> innerME = secME[sector_id];
	      MonitorElement* globaleff = innerME.find("TrigEff_Theta")->second;
	      float bineff = float(Track_thpos_andtrig_histo->GetEntries())/Track_thpos_histo->GetEntries();
	      float binerr = sqrt(bineff*(1-bineff)/Track_thpos_histo->GetEntries());
	      globaleff->setBinContent(stat,bineff);
	      globaleff->setBinError(stat,binerr);
	      DTChamberId dtChId(wh,stat,sect);
	      uint32_t indexCh = dtChId.rawId();
	      if( chambME[indexCh].find("TrigEffAngle_Theta") == chambME[indexCh].end()){
		bookChambHistos(dtChId,"TrigEffPos_Theta");
		bookChambHistos(dtChId,"TrigEffAngle_Theta");
	      }
	      innerME = chambME[indexCh];
	      makeEfficiencyME(Track_thpos_andtrig_histo,Track_thpos_histo,innerME.find("TrigEffPos_Theta")->second);
	      makeEfficiencyME(Track_thangle_andtrig_histo,Track_thangle_histo,innerME.find("TrigEffAngle_Theta")->second);

	    }
	  }
	}
      }
    }
  }
  
  // Efficiency test (performed on chamber plots)
  for(map<uint32_t,map<string,MonitorElement*> >::const_iterator imapIt = chambME.begin();
      imapIt != chambME.end();
      ++imapIt){
    for (map<string,MonitorElement*>::const_iterator effME = (*imapIt).second.begin();
	 effME!=(*imapIt).second.end();
	 ++effME){
      if ((*effME).second->getName().find("TrigEffPos_Phi") == 0) {
	const QReport *effQReport = (*effME).second->getQReport("ChambTrigEffInRangePhi");
	if (effQReport) {
	  if (effQReport->getBadChannels().size())
	    edm::LogError ("localTrigger") << (*effME).second->getName() <<" has " << effQReport->getBadChannels().size() << " channels out of expected efficiency range";
	  edm::LogWarning ("localTrigger") << "-------" << effQReport->getMessage() << " ------- " << effQReport->getStatus();
	}
      }
      if ((*effME).second->getName().find("TrigEffPos_Theta") == 0) {
	const QReport *effQReport = (*effME).second->getQReport("ChambTrigEffInRangeTheta");
	if (effQReport) {
	  if (effQReport->getBadChannels().size())
	    edm::LogError ("localTrigger") << (*effME).second->getName() <<" has " << effQReport->getBadChannels().size() << " channels out of expected efficiency range";
	  edm::LogWarning ("localTrigger") << "-------" << effQReport->getMessage() << " ------- " << effQReport->getStatus();
	}
      }
    }
  }

  // Efficiency test (performed on wheel plots)
  for(map<int,map<string,MonitorElement*> >::const_iterator imapIt = secME.begin();
      imapIt != secME.end();
      ++imapIt){
    for (map<string,MonitorElement*>::const_iterator effME = (*imapIt).second.begin();
	 effME!=(*imapIt).second.end();
	 ++effME){
      if ((*effME).second->getName().find("TrigEff_Phi") == 0) {
	const QReport *effQReport = (*effME).second->getQReport("SectorTrigEffInRangePhi");
	if (effQReport) {
	  edm::LogWarning ("localTrigger") << "-------" << effQReport->getMessage() << " ------- " << effQReport->getStatus();
	}
      }
      if ((*effME).second->getName().find("TrigEff_Theta") == 0) {
	const QReport *effQReport = (*effME).second->getQReport("SectorTrigEffInRangeTheta");
	if (effQReport) {
	  edm::LogWarning ("localTrigger") << "-------" << effQReport->getMessage() << " ------- " << effQReport->getStatus();
	}
      }
    }
  }

  if (nevents%parameters.getUntrackedParameter<int>("resultsSavingRate",10) == 0){
    if ( parameters.getUntrackedParameter<bool>("writeHisto", true) ) 
      dbe->save(parameters.getUntrackedParameter<string>("outputFile", "DTLocalTriggerTest.root"));
  }

}

void DTLocalTriggerTest::makeEfficiencyME(TH1F* numerator, TH1F* denominator, MonitorElement* result){
  
  MonitorElementT<TNamed>* efficiencyME = dynamic_cast<MonitorElementT<TNamed>*>(result);
  TH1F* efficiency = dynamic_cast<TH1F*> (efficiencyME->operator->());
  efficiency->Divide(numerator,denominator,1,1,"");
  
  int nbins = efficiency->GetNbinsX();
  for (int bin=1; bin<=nbins; ++bin){
    float error = 0;
    float bineff = efficiency->GetBinContent(bin);

    if (denominator->GetBinContent(bin)){
      error = sqrt(bineff*(1-bineff)/denominator->GetBinContent(bin));
    }
    else {
      error = 1;
      efficiency->SetBinContent(bin,1.);
    }
 
    efficiency->SetBinError(bin,error);
  }

}
    
string DTLocalTriggerTest::getMEName(string histoTag, string subfolder, const DTChamberId & chambid) {

  stringstream wheel; wheel << chambid.wheel();
  stringstream station; station << chambid.station();
  stringstream sector; sector << chambid.sector();

  if (subfolder == "Segment" && histoTag.find("Trig") == string::npos) 
    histoTag = "SEG_" + histoTag;
  else
    histoTag = hwSource + "_" + histoTag;

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
