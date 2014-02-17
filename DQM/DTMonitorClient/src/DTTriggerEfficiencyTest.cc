/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/10/28 08:12:44 $
 *  $Revision: 1.6 $
 *  \author C. Battilana - CIEMAT
 */


// This class header
#include "DQM/DTMonitorClient/src/DTTriggerEfficiencyTest.h"

// Framework headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

// Trigger
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

// Root
#include "TF1.h"
#include "TProfile.h"


//C++ headers
#include <iostream>
#include <sstream>

using namespace edm;
using namespace std;


DTTriggerEfficiencyTest::DTTriggerEfficiencyTest(const edm::ParameterSet& ps){

  setConfig(ps,"DTTriggerEfficiency");
  baseFolderDCC = "DT/03-LocalTrigger-DCC/";
  baseFolderDDU = "DT/04-LocalTrigger-DDU/";
  detailedPlots = ps.getUntrackedParameter<bool>("detailedAnalysis",true);

}


DTTriggerEfficiencyTest::~DTTriggerEfficiencyTest(){

}


void DTTriggerEfficiencyTest::beginJob(){

  DTLocalTriggerBaseTest::beginJob();

}


void DTTriggerEfficiencyTest::beginRun(const edm::Run& r,const edm::EventSetup& c){

  DTLocalTriggerBaseTest::beginRun(r,c);
  trigGeomUtils = new DTTrigGeomUtils(muonGeom);

  vector<string>::const_iterator iTr   = trigSources.begin();
  vector<string>::const_iterator trEnd = trigSources.end();
  vector<string>::const_iterator iHw   = hwSources.begin();
  vector<string>::const_iterator hwEnd = hwSources.end();


  //Booking
  if(parameters.getUntrackedParameter<bool>("staticBooking", true)){
    for (; iTr != trEnd; ++iTr){
      trigSource = (*iTr);
      for (; iHw != hwEnd; ++iHw){
        hwSource = (*iHw);
        // Loop over the TriggerUnits
        bookHistos("TrigEffPhi","");
        bookHistos("TrigEffCorrPhi","");
        for (int wh=-2; wh<=2; ++wh){
          if (detailedPlots) {
            for (int sect=1; sect<=12; ++sect){
              for (int stat=1; stat<=4; ++stat){
                DTChamberId chId(wh,stat,sect);
                bookChambHistos(chId,"TrigEffPosvsAnglePhi","Segment");
                bookChambHistos(chId,"TrigEffPosvsAngleCorrPhi","Segment");
              }
            }
          }
          bookWheelHistos(wh,"TrigEffPhi","");  
          bookWheelHistos(wh,"TrigEffCorrPhi","");  
        }
      }
    }
  }

}


void DTTriggerEfficiencyTest::runClientDiagnostic() {

  // Loop over Trig & Hw sources
  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr){
    trigSource = (*iTr);
    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw){
      hwSource = (*iHw);
      // Loop over the TriggerUnits
      if( globalEffDistr.find(fullName("TrigEffPhi")) == globalEffDistr.end() ){
        bookHistos("TrigEffPhi","");
        bookHistos("TrigEffCorrPhi","");
      }
      for (int wh=-2; wh<=2; ++wh){

        TH2F * TrigEffDenum   = getHisto<TH2F>(dbe->get(getMEName("TrigEffDenum","Task",wh)));
        TH2F * TrigEffNum     = getHisto<TH2F>(dbe->get(getMEName("TrigEffNum","Task",wh)));
        TH2F * TrigEffCorrNum = getHisto<TH2F>(dbe->get(getMEName("TrigEffCorrNum","Task",wh)));

        if (TrigEffDenum && TrigEffNum && TrigEffCorrNum && TrigEffDenum->GetEntries()>1) {

          if( whME[wh].find(fullName("TrigEffPhi")) == whME[wh].end() ){
            bookWheelHistos(wh,"TrigEffPhi","");  
            bookWheelHistos(wh,"TrigEffCorrPhi","");  
          }

          MonitorElement* Eff1DAll_TrigEffPhi = (&globalEffDistr)->find(fullName("TrigEffPhi"))->second;
          MonitorElement* Eff1DAll_TrigEffCorrPhi = (&globalEffDistr)->find(fullName("TrigEffCorrPhi"))->second;

          MonitorElement* Eff1DWh_TrigEffPhi = (&(EffDistrPerWh[wh]))->find(fullName("TrigEffPhi"))->second;
          MonitorElement* Eff1DWh_TrigEffCorrPhi = (&(EffDistrPerWh[wh]))->find(fullName("TrigEffCorrPhi"))->second;

          MonitorElement* Eff2DWh_TrigEffPhi = (&(whME[wh]))->find(fullName("TrigEffPhi"))->second;
          MonitorElement* Eff2DWh_TrigEffCorrPhi = (&(whME[wh]))->find(fullName("TrigEffCorrPhi"))->second;

          makeEfficiencyME(TrigEffNum,TrigEffDenum,Eff2DWh_TrigEffPhi,Eff1DWh_TrigEffPhi,Eff1DAll_TrigEffPhi);
          makeEfficiencyME(TrigEffCorrNum,TrigEffDenum,Eff2DWh_TrigEffCorrPhi,Eff1DWh_TrigEffCorrPhi,Eff1DAll_TrigEffCorrPhi);

        }

        if (detailedPlots) {
          for (int stat=1; stat<=4; ++stat){
            for (int sect=1; sect<=12; ++sect){
              DTChamberId chId(wh,stat,sect);
              uint32_t indexCh = chId.rawId();

              // Perform Efficiency analysis (Phi+Segments 2D)
              TH2F * TrackPosvsAngle        = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngle","Segment", chId)));
              TH2F * TrackPosvsAngleAnyQual = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngleAnyQual","Segment", chId)));
              TH2F * TrackPosvsAngleCorr    = getHisto<TH2F>(dbe->get(getMEName("TrackPosvsAngleCorr","Segment", chId)));

              if (TrackPosvsAngle && TrackPosvsAngleAnyQual && TrackPosvsAngleCorr && TrackPosvsAngle->GetEntries()>1) {

                if( chambME[indexCh].find(fullName("TrigEffAnglePhi")) == chambME[indexCh].end()){
                  bookChambHistos(chId,"TrigEffPosvsAnglePhi","Segment");
                  bookChambHistos(chId,"TrigEffPosvsAngleCorrPhi","Segment");
                }

                std::map<std::string,MonitorElement*> *innerME = &(chambME[indexCh]);
                makeEfficiencyME(TrackPosvsAngleAnyQual,TrackPosvsAngle,innerME->find(fullName("TrigEffPosvsAnglePhi"))->second);
                makeEfficiencyME(TrackPosvsAngleCorr,TrackPosvsAngle,innerME->find(fullName("TrigEffPosvsAngleCorrPhi"))->second);

              }
            }
          }
        }
      }

    }
  }	

}

void DTTriggerEfficiencyTest::makeEfficiencyME(TH2F* numerator, TH2F* denominator, MonitorElement* result2DWh, MonitorElement* result1DWh, MonitorElement* result1D){

  TH2F* efficiency = result2DWh->getTH2F();
  efficiency->Divide(numerator,denominator,1,1,"");

  int nbinsx = efficiency->GetNbinsX();
  int nbinsy = efficiency->GetNbinsY();
  for (int binx=1; binx<=nbinsx; ++binx){
    for (int biny=1; biny<=nbinsy; ++biny){
      float error = 0;
      float bineff = efficiency->GetBinContent(binx,biny);

      result1DWh->Fill(bineff);
      result1D->Fill(bineff);

      if (denominator->GetBinContent(binx,biny)){
        error = sqrt(bineff*(1-bineff)/denominator->GetBinContent(binx,biny));
      }
      else {
        error = 1;
        efficiency->SetBinContent(binx,biny,0.);
      }

      efficiency->SetBinError(binx,biny,error);
    }
  }

}

void DTTriggerEfficiencyTest::makeEfficiencyME(TH2F* numerator, TH2F* denominator, MonitorElement* result2DWh){

  TH2F* efficiency = result2DWh->getTH2F();
  efficiency->Divide(numerator,denominator,1,1,"");

  int nbinsx = efficiency->GetNbinsX();
  int nbinsy = efficiency->GetNbinsY();
  for (int binx=1; binx<=nbinsx; ++binx){
    for (int biny=1; biny<=nbinsy; ++biny){
      float error = 0;
      float bineff = efficiency->GetBinContent(binx,biny);

      if (denominator->GetBinContent(binx,biny)){
        error = sqrt(bineff*(1-bineff)/denominator->GetBinContent(binx,biny));
      }
      else {
        error = 1;
        efficiency->SetBinContent(binx,biny,0.);
      }

      efficiency->SetBinError(binx,biny,error);
    }
  }

}    

string DTTriggerEfficiencyTest::getMEName(string histoTag, string folder, int wh) {

  stringstream wheel; wheel << wh;

  string folderName =  topFolder(hwSource=="DCC") + folder + "/";

  string histoname = sourceFolder + folderName 
    + fullName(histoTag) + "_W" + wheel.str();

  return histoname;

}

void DTTriggerEfficiencyTest::bookHistos(string hTag,string folder) {

  string basedir;  
  bool isDCC = hwSource=="DCC" ;  
  basedir = topFolder(isDCC);   //Book summary histo outside Task directory 

  if (folder != "") {
    basedir += folder +"/" ;
  }
  dbe->setCurrentFolder(basedir);

  string fullTag = fullName(hTag);
  string hname = fullTag + "_All";

  globalEffDistr[fullTag] = dbe->book1D(hname.c_str(),hname.c_str(),51,0.,1.02);
  globalEffDistr[fullTag] ->setAxisTitle("Trig Eff",1);

}

void DTTriggerEfficiencyTest::bookWheelHistos(int wheel,string hTag,string folder) {

  stringstream wh; wh << wheel;
  string basedir;  
  bool isDCC = hwSource=="DCC" ;  
  if (hTag.find("Summary") != string::npos) {
    basedir = topFolder(isDCC);   //Book summary histo outside wheel directories
  } else {
    basedir = topFolder(isDCC) + "Wheel" + wh.str() + "/" ;

  }
  if (folder != "") {
    basedir += folder +"/" ;
  }
  dbe->setCurrentFolder(basedir);

  string fullTag = fullName(hTag);
  string hname    = fullTag+ "_W" + wh.str();

  string hnameAll = fullTag+ "_All_W" + wh.str();

  LogTrace(category()) << "[" << testName << "Test]: booking "<< basedir << hname;

  (EffDistrPerWh[wheel])[fullTag] = dbe->book1D(hnameAll.c_str(),hnameAll.c_str(),51,0.,1.02);

  if (hTag.find("Phi")!= string::npos ||
      hTag.find("Summary") != string::npos ){    
    MonitorElement* me = dbe->book2D(hname.c_str(),hname.c_str(),12,1,13,4,1,5);

    //     setLabelPh(me);
    me->setBinLabel(1,"MB1",2);
    me->setBinLabel(2,"MB2",2);
    me->setBinLabel(3,"MB3",2);
    me->setBinLabel(4,"MB4",2);
    me->setAxisTitle("Sector",1);

    whME[wheel][fullTag] = me;
    return;
  }

  if (hTag.find("Theta") != string::npos){
    MonitorElement* me =dbe->book2D(hname.c_str(),hname.c_str(),12,1,13,3,1,4);

    //     setLabelTh(me);
    me->setBinLabel(1,"MB1",2);
    me->setBinLabel(2,"MB2",2);
    me->setBinLabel(3,"MB3",2);
    me->setAxisTitle("Sector",1);

    whME[wheel][fullTag] = me;
    return;
  }

}

void DTTriggerEfficiencyTest::bookChambHistos(DTChamberId chambId, string htype, string folder) {

  stringstream wheel; wheel << chambId.wheel();
  stringstream station; station << chambId.station();	
  stringstream sector; sector << chambId.sector();

  string fullType  = fullName(htype);
  bool isDCC = hwSource=="DCC" ;
  string HistoName = fullType + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  dbe->setCurrentFolder(topFolder(isDCC) + 
      "Wheel" + wheel.str() +
      "/Sector" + sector.str() +
      "/Station" + station.str() + 
      "/" + folder + "/");

  LogTrace(category()) << "[" << testName << "Test]: booking " + topFolder(isDCC) + "Wheel" << wheel.str() 
    <<"/Sector" << sector.str() << "/Station" << station.str() << "/" + folder + "/" << HistoName;


  uint32_t indexChId = chambId.rawId();
  float min, max;
  int nbins;
  trigGeomUtils->phiRange(chambId,min,max,nbins,20);
  if (htype.find("TrigEffPosvsAnglePhi") == 0 ){
    chambME[indexChId][fullType] = dbe->book2D(HistoName.c_str(),"Trigger efficiency (any qual.) position vs angle (Phi)",12,-30.,30.,nbins,min,max);
    return;
  }
  if (htype.find("TrigEffPosvsAngleCorrPhi") == 0 ){
    chambME[indexChId][fullType] = dbe->book2D(HistoName.c_str(),"Trigger efficiency (correlated) pos vs angle (Phi)",12,-30.,30.,nbins,min,max);
    return;
  }

}
