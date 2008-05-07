#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripSummaryCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <iostream>
#include <iomanip>
using namespace std;
//
// -- Constructor
// 
SiStripActionExecutor::SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Creating SiStripActionExecutor " << "\n" ;
  summaryCreator_= 0;
  tkMapCreator_ = 0; 
}
//
// --  Destructor
// 
SiStripActionExecutor::~SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Deleting SiStripActionExecutor " << "\n" ;
  if (summaryCreator_) delete   summaryCreator_;
  if (tkMapCreator_) delete   tkMapCreator_;
}
//
// -- Read Configurationn File
//
bool SiStripActionExecutor::readConfiguration() {
  
  if (!summaryCreator_) {
    summaryCreator_ = new SiStripSummaryCreator();
  }
  if (summaryCreator_->readConfiguration()) return true;
  else return false;
}
//
// -- Read Configurationn File
//
bool SiStripActionExecutor::readTkMapConfiguration() {
  
  if (tkMapCreator_) delete tkMapCreator_;
  tkMapCreator_ = new SiStripTrackerMapCreator();
  if (tkMapCreator_->readConfiguration()) return true;
  else return false;
}
//
// -- Read Configurationn File
//
bool SiStripActionExecutor::readConfiguration(int& sum_freq) {
  bool result = false;
  if (readConfiguration()) {
    sum_freq = summaryCreator_->getFrequency();
    if (sum_freq != -1) result = true;
  }
  return result;
}
//
// -- Create and Fill Summary Monitor Elements
//
void SiStripActionExecutor::createSummary(DQMStore* dqm_store) {
  if (summaryCreator_) {
    dqm_store->cd();
    string dname = "SiStrip/MechanicalView";
    if (dqm_store->dirExists(dname)) {
      dqm_store->cd(dname);
      summaryCreator_->createSummary(dqm_store);
    }
  }
}
//
// -- create tracker map
//
void SiStripActionExecutor::createTkMap(const edm::ParameterSet & tkmapPset, 
           const edm::ESHandle<SiStripFedCabling>& fedcabling, DQMStore* dqm_store) {
  if (tkMapCreator_) tkMapCreator_->create(tkmapPset, fedcabling, dqm_store);
}
//
// -- create reportSummary MEs
//
void SiStripActionExecutor::bookGlobalStatus(DQMStore* dqm_store) {

  dqm_store->cd();

  dqm_store->setCurrentFolder("SiStrip/EventInfo");    
  SummaryReport = dqm_store->bookFloat("reportSummary");
 
  SummaryReportMap = dqm_store->book2D("reportSummaryMap","SiStrip Report Summary Map",6,0.5,6.5,10,0.5,10.5);
  SummaryReportMap->setAxisTitle("Sub Detector Trype", 1);
  SummaryReportMap->setAxisTitle("Layer/Disc Number", 2);
  SummaryReportMap->setBinLabel(1, "TIB");
  SummaryReportMap->setBinLabel(2, "TOB");
  SummaryReportMap->setBinLabel(3, "TIDF");
  SummaryReportMap->setBinLabel(4, "TIDB");
  SummaryReportMap->setBinLabel(5, "TECF");
  SummaryReportMap->setBinLabel(6, "TECB");
  
  dqm_store->setCurrentFolder("SiStrip/EventInfo/reportSummaryContents");      
  
  SummaryTIB  = dqm_store->bookFloat("SummaryTIB");
  SummaryTOB  = dqm_store->bookFloat("SummaryTOB");
  SummaryTIDF = dqm_store->bookFloat("SummaryTIDF");
  SummaryTIDB = dqm_store->bookFloat("SummaryTIDB");
  SummaryTECF = dqm_store->bookFloat("SummaryTECF");
  SummaryTECB = dqm_store->bookFloat("SummaryTECB");

}
// 
// -- Fill Global Status
//
void SiStripActionExecutor::fillGlobalStatus(const edm::ESHandle<SiStripDetCabling>& detcabling, DQMStore* dqm_store) {
  float gStatus = 0.0;
  // get connected detectors
  std::vector<uint32_t> SelectedDetIds;
  detcabling->addActiveDetectorsRawIds(SelectedDetIds);
  int nDetErr = 0;
  int nDetTot = 0;
  int nDetTIBErr, nDetTOBErr, nDetTIDFErr, nDetTIDBErr, nDetTECFErr, nDetTECBErr;
  int nDetTIBTot, nDetTOBTot, nDetTIDFTot, nDetTIDBTot, nDetTECFTot, nDetTECBTot;
  float statusTIB, statusTOB,  statusTIDF,  statusTIDB,  statusTECF,  statusTECB;

  statusTIB = statusTOB = statusTIDF = statusTIDB = statusTECF = statusTECB = -1;
  nDetTIBErr = nDetTOBErr = nDetTIDFErr = nDetTIDBErr = nDetTECFErr = nDetTECBErr = 0;
  nDetTIBTot = nDetTOBTot = nDetTIDFTot = nDetTIDBTot = nDetTECFTot = nDetTECBTot = 0;
  SiStripFolderOrganizer folder_organizer;
  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(), iEnd=SelectedDetIds.end();idetid!=iEnd;++idetid){    
    uint32_t detId = *idetid;
    if (detId == 0 || detId == 0xFFFFFFFF){
      edm::LogError("SiStripAnalyser") 
                          << "SiStripAnalyser::fillGlobalStatus : " 
                          << "Wrong DetId !!!!!! " <<  detId << " Neglecting !!!!!! ";
      continue;
    }
    StripSubdetector subdet(*idetid);
    string dir_path;
    folder_organizer.getFolderName(detId, dir_path);     
    vector<MonitorElement*> detector_mes = dqm_store->getContents(dir_path);
    int error_me = 0;
    for (vector<MonitorElement *>::const_iterator it = detector_mes.begin();
	 it!= detector_mes.end(); it++) {
      MonitorElement * me = (*it);     
      if (!me) continue;
      if (me->getQReports().size() == 0) continue;
      int istat =  SiStripUtility::getMEStatus((*it)); 
      if (istat == dqm::qstatus::ERROR)  error_me++;
    }
    nDetTot++;
        
    if (error_me > 0) {
     nDetErr++;
    }
    switch (subdet.subdetId()) 
      {
      case StripSubdetector::TIB:
	{
	  nDetTIBTot++;
	  if (error_me > 0) nDetTIBErr++;
	  break;       
	}
      case StripSubdetector::TID:
	{
	  TIDDetId tidId(detId);
	  if (tidId.side() == 2) {
	    nDetTIDFTot++;
	    if (error_me > 0) nDetTIDFErr++;
	  }  else if (tidId.side() == 1) {
	    nDetTIDBTot++;
	    if (error_me > 0) nDetTIDBErr++;
	  }
	  break;       
	}
      case StripSubdetector::TOB:
	{
	  nDetTOBTot++;
	  if (error_me > 0) nDetTOBErr++;
	  break;       
	}
      case StripSubdetector::TEC:
	{
	  TECDetId tecId(detId);
	  if (tecId.side() == 2) {
	    nDetTECFTot++;
	    if (error_me > 0) nDetTECFErr++;
	  }  else if (tecId.side() == 1) {
	    nDetTECBTot++;
	    if (error_me > 0) nDetTECBErr++;
	  }
	  break;       
	}
      }
  }
  gStatus = (1 - nDetErr*1.0/nDetTot);
  SummaryReport->Fill(gStatus);

  if (nDetTIBTot  > 0) statusTIB  = (1 - nDetTIBErr*1.0/nDetTIBTot);
  if (nDetTOBTot  > 0) statusTOB  = (1 - nDetTOBErr*1.0/nDetTOBTot);
  if (nDetTIDFTot > 0) statusTIDF = (1 - nDetTIDFErr*1.0/nDetTIDFTot);
  if (nDetTIDBTot > 0) statusTIDB = (1 - nDetTIDBErr*1.0/nDetTIDBTot);
  if (nDetTECFTot > 0) statusTECF = (1 - nDetTECFErr*1.0/nDetTECFTot);
  if (nDetTECBTot > 0) statusTECB = (1 - nDetTECBErr*1.0/nDetTECBTot);

  

  SummaryTIB->Fill(statusTIB);
  SummaryTOB->Fill(statusTOB);
  SummaryTIDF->Fill(statusTIDF);
  SummaryTIDB->Fill(statusTIDB);
  SummaryTECF->Fill(statusTECF);
  SummaryTECB->Fill(statusTECB);
  
  cout <<"# of Det TIB : (tot)"<<setw(5)<<nDetTIBTot<< " (error) "<<nDetTIBErr <<" ==> "<<statusTIB<< endl; 
  cout <<"# of Det TOB : (tot)"<<setw(5)<<nDetTOBTot<< " (error) "<<nDetTOBErr <<" ==> "<<statusTOB<< endl; 
  cout <<"# of Det TIDF: (tot)"<<setw(5)<<nDetTIDFTot<<" (error) "<<nDetTIDFErr<<" ==> "<<statusTIDF<< endl; 
  cout <<"# of Det TIDB: (tot)"<<setw(5)<<nDetTIDBTot<<" (error) "<<nDetTIDBErr<<" ==> "<<statusTIDB<< endl; 
  cout <<"# of Det TECF: (tot)"<<setw(5)<<nDetTECFTot<<" (error) "<<nDetTECFErr<<" ==> "<<statusTECF<< endl; 
  cout <<"# of Det TECB: (tot)"<<setw(5)<<nDetTECBTot<<" (error) "<<nDetTECBErr<<" ==> "<<statusTECB<< endl; 

  SummaryReportMap->Reset();
  string dname;
  dname = "SiStrip/MechanicalView/TIB";
  fillSubDetStatus(dqm_store, dname, 1);  
  dname = "SiStrip/MechanicalView/TOB";
  fillSubDetStatus(dqm_store, dname, 2);  
  dname = "SiStrip/MechanicalView/TID/side_2";
  fillSubDetStatus(dqm_store, dname, 3);  
  dname = "SiStrip/MechanicalView/TID/side_1";
  fillSubDetStatus(dqm_store, dname, 4);  
  dname = "SiStrip/MechanicalView/TEC/side_2";
  fillSubDetStatus(dqm_store, dname, 5);  
  dname = "SiStrip/MechanicalView/TEC/side_1";
  fillSubDetStatus(dqm_store, dname, 6);  
}
//
// -- fill subDetStatus
//
void SiStripActionExecutor::fillSubDetStatus(DQMStore* dqm_store, string& dname, int xbin) {
  if (dqm_store->dirExists(dname)) {
    dqm_store->cd(dname);
    vector<string> subDirVec = dqm_store->getSubdirs();
    int ybin = 0;     
    for (vector<string>::const_iterator ic = subDirVec.begin();
	 ic != subDirVec.end(); ic++) {
      vector<MonitorElement*> meVec;
      meVec = dqm_store->getAllContents((*ic));
      float error_me = 0.0;
      float tot_me = 0.0;
      for (vector<MonitorElement*>::const_iterator it = meVec.begin();
               it != meVec.end(); it++) {
	MonitorElement * me = (*it);     
	if (!me) continue;
        tot_me++;
	if (me->getQReports().size() == 0) continue;
	int istat =  SiStripUtility::getMEStatus((*it)); 
	if (istat == dqm::qstatus::ERROR)  error_me++;
      }
      ybin++;
      if (tot_me > 0.0) SummaryReportMap->Fill(xbin,ybin, (1-error_me/tot_me));
      else SummaryReportMap->Fill(xbin,ybin, -1.0);
    }
  }  
}
//
// -- create reportSummary MEs
//
void SiStripActionExecutor::resetGlobalStatus() {

  SummaryReport->Reset();
 
  SummaryReportMap->Reset();

  SummaryTIB->Reset();
  SummaryTOB->Reset();
  SummaryTIDF->Reset();
  SummaryTIDB->Reset();
  SummaryTECF->Reset();
  SummaryTECB->Reset();
}
