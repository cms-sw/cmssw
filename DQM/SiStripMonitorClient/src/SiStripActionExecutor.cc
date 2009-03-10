
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

  bookedGlobalStatus_ = false;
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
// -- Create and Fill Summary Monitor Elements
//
void SiStripActionExecutor::createSummaryOffline(DQMStore* dqm_store) {
  if (summaryCreator_) {
    dqm_store->cd();
    string dname = "MechanicalView";
    if (goToDir(dqm_store, dname)) {
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

  if (!bookedGlobalStatus_) {
    dqm_store->cd();
    
    dqm_store->setCurrentFolder("SiStrip/EventInfo");    
    SummaryReport = dqm_store->bookFloat("reportSummary");
    
    SummaryReportMap = dqm_store->book2D("reportSummaryMap","SiStrip Report Summary Map",6,0.5,6.5,9,0.5,9.5);
    SummaryReportMap->setAxisTitle("Sub Detector Type", 1);
    SummaryReportMap->setAxisTitle("Layer/Disc Number", 2);
    SummaryReportMap->setBinLabel(1, "TIB");
    SummaryReportMap->setBinLabel(2, "TOB");
    SummaryReportMap->setBinLabel(3, "TIDF");
    SummaryReportMap->setBinLabel(4, "TIDB");
    SummaryReportMap->setBinLabel(5, "TECF");
    SummaryReportMap->setBinLabel(6, "TECB");
    
    dqm_store->setCurrentFolder("SiStrip/EventInfo/reportSummaryContents");      
    
    SummaryTIB  = dqm_store->bookFloat("SiStrip_TIB");
    SummaryTOB  = dqm_store->bookFloat("SiStrip_TOB");
    SummaryTIDF = dqm_store->bookFloat("SiStrip_TIDF");
    SummaryTIDB = dqm_store->bookFloat("SiStrip_TIDB");
    SummaryTECF = dqm_store->bookFloat("SiStrip_TECF");
    SummaryTECB = dqm_store->bookFloat("SiStrip_TECB");

    dqm_store->setCurrentFolder("SiStrip/Tracks");      
    OnTrackClusterReport = dqm_store->book1D("OnTrackClustersReport", "OnTrackClusterReport",34,0.5,34.5);
    OnTrackClusterReport->setAxisTitle("# of On Track Clusters", 2);
    OnTrackClusterReport->setBinLabel(1,"TIB_L1");
    OnTrackClusterReport->setBinLabel(2,"TIB_L2");
    OnTrackClusterReport->setBinLabel(3,"TIB_L3");
    OnTrackClusterReport->setBinLabel(4,"TOB_L4");
    OnTrackClusterReport->setBinLabel(5,"TOB_L1");
    OnTrackClusterReport->setBinLabel(6,"TOB_L2");
    OnTrackClusterReport->setBinLabel(7,"TOB_L3");
    OnTrackClusterReport->setBinLabel(8,"TOB_L4");
    OnTrackClusterReport->setBinLabel(9,"TOB_L5");
    OnTrackClusterReport->setBinLabel(10,"TOB_L6");
    OnTrackClusterReport->setBinLabel(11,"TIDF_W1");
    OnTrackClusterReport->setBinLabel(12,"TIDF_W2");
    OnTrackClusterReport->setBinLabel(13,"TIDF_W2");
    OnTrackClusterReport->setBinLabel(14,"TIDB_W1");
    OnTrackClusterReport->setBinLabel(15,"TIDB_W2");
    OnTrackClusterReport->setBinLabel(16,"TIDB_W2");
    OnTrackClusterReport->setBinLabel(17,"TECF_W1");
    OnTrackClusterReport->setBinLabel(18,"TECF_W2");
    OnTrackClusterReport->setBinLabel(19,"TECF_W3");
    OnTrackClusterReport->setBinLabel(20,"TECF_W4");
    OnTrackClusterReport->setBinLabel(21,"TECF_W5");
    OnTrackClusterReport->setBinLabel(22,"TECF_W6");
    OnTrackClusterReport->setBinLabel(23,"TECF_W7");
    OnTrackClusterReport->setBinLabel(24,"TECF_W8");
    OnTrackClusterReport->setBinLabel(25,"TECF_W9");
    OnTrackClusterReport->setBinLabel(26,"TECB_W1");
    OnTrackClusterReport->setBinLabel(27,"TECB_W2");
    OnTrackClusterReport->setBinLabel(28,"TECB_W3");
    OnTrackClusterReport->setBinLabel(29,"TECB_W4");
    OnTrackClusterReport->setBinLabel(30,"TECB_W5");
    OnTrackClusterReport->setBinLabel(31,"TECB_W6");
    OnTrackClusterReport->setBinLabel(32,"TECB_W7");
    OnTrackClusterReport->setBinLabel(33,"TECB_W8");
    OnTrackClusterReport->setBinLabel(34,"TECB_W9");
    
    bookedGlobalStatus_ = true;
    fillDummyGlobalStatus();
  }
}
//
// -- Fill Dummy Global Status
//
void SiStripActionExecutor::fillDummyGlobalStatus(){
  
  resetGlobalStatus();

  SummaryReport->Fill(-1.0);

  SummaryTIB->Fill(-1.0);
  SummaryTOB->Fill(-1.0);
  SummaryTIDF->Fill(-1.0);
  SummaryTIDB->Fill(-1.0);
  SummaryTECF->Fill(-1.0);
  SummaryTECB->Fill(-1.0);
  
  for (unsigned int xbin = 1; xbin < 7; xbin++) {
    for (unsigned int ybin = 1; ybin < 10; ybin++) {
      SummaryReportMap->Fill(xbin, ybin, -1.0);
    }
  }
}
// 
// -- Fill Global Status
//
void SiStripActionExecutor::fillGlobalStatusFromModule(DQMStore* dqm_store) {
  if (!bookedGlobalStatus_) bookGlobalStatus(dqm_store);
  float gStatus = -1.0;
  int nDetErr = 0;
  int nDetTot = 0;
  int nDetTIBErr, nDetTOBErr, nDetTIDFErr, nDetTIDBErr, nDetTECFErr, nDetTECBErr;
  int nDetTIBTot, nDetTOBTot, nDetTIDFTot, nDetTIDBTot, nDetTECFTot, nDetTECBTot;
  float statusTIB, statusTOB,  statusTIDF,  statusTIDB,  statusTECF,  statusTECB;

  statusTIB  = statusTOB  = statusTIDF  = statusTIDB  = statusTECF  = statusTECB  = -1;
  nDetTIBErr = nDetTOBErr = nDetTIDFErr = nDetTIDBErr = nDetTECFErr = nDetTECBErr = 0;
  nDetTIBTot = nDetTOBTot = nDetTIDFTot = nDetTIDBTot = nDetTECFTot = nDetTECBTot = 0;

  fillDummyGlobalStatus();
  string dname;
  // Get Status for TIB
  dname = "SiStrip/MechanicalView/TIB";
  fillSubDetStatusFromModule(dqm_store, dname, nDetTIBTot, nDetTIBErr, 1);
  fillClusterReport(dqm_store, dname, 0);
  // Get Status for TOB
  dname = "SiStrip/MechanicalView/TOB";
  fillSubDetStatusFromModule(dqm_store, dname, nDetTOBTot, nDetTOBErr, 2);  
  fillClusterReport(dqm_store, dname, 4);
  // Get Status for TIDF
  dname = "SiStrip/MechanicalView/TID/side_2";
  fillSubDetStatusFromModule(dqm_store, dname, nDetTIDFTot, nDetTIDFErr, 3);  
  fillClusterReport(dqm_store, dname, 10);
  // Get Status for TIDB 
  dname = "SiStrip/MechanicalView/TID/side_1";
  fillSubDetStatusFromModule(dqm_store, dname, nDetTIDBTot, nDetTIDBErr, 4);  
  fillClusterReport(dqm_store, dname, 13);
  // Get Status for TECF 
  dname = "SiStrip/MechanicalView/TEC/side_2";
  fillSubDetStatusFromModule(dqm_store, dname, nDetTECFTot, nDetTECFErr, 5);  
  fillClusterReport(dqm_store, dname, 16);
  // Get Status for TECB
  dname = "SiStrip/MechanicalView/TEC/side_1";
  fillSubDetStatusFromModule(dqm_store, dname, nDetTECBTot, nDetTECBErr, 6);  
  fillClusterReport(dqm_store, dname, 25);

  nDetTot = nDetTIBTot + nDetTOBTot + nDetTIDFTot + nDetTIDBTot + nDetTECFTot + nDetTECBTot;
  nDetErr = nDetTIBErr + nDetTOBErr + nDetTIDFErr + nDetTIDBErr + nDetTECFErr + nDetTECBErr;
  if (nDetTot > 0) gStatus = (1 - nDetErr*1.0/nDetTot);
  
  if (nDetTIBTot  > 0) statusTIB  = (1 - nDetTIBErr*1.0/nDetTIBTot);
  if (nDetTOBTot  > 0) statusTOB  = (1 - nDetTOBErr*1.0/nDetTOBTot);
  if (nDetTIDFTot > 0) statusTIDF = (1 - nDetTIDFErr*1.0/nDetTIDFTot);
  if (nDetTIDBTot > 0) statusTIDB = (1 - nDetTIDBErr*1.0/nDetTIDBTot);
  if (nDetTECFTot > 0) statusTECF = (1 - nDetTECFErr*1.0/nDetTECFTot);
  if (nDetTECBTot > 0) statusTECB = (1 - nDetTECBErr*1.0/nDetTECBTot);
  
  SummaryReport->Fill(gStatus);
    
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
}
// 
// -- Fill Global Status for Tier0
//
void SiStripActionExecutor::fillGlobalStatusFromLayer(DQMStore* dqm_store) {
  if (!bookedGlobalStatus_) bookGlobalStatus(dqm_store);
  float gStatus = -1.0;
  int nDetErr = 0;
  int nDetTot = 0;
  int nDetTIBErr, nDetTOBErr, nDetTIDFErr, nDetTIDBErr, nDetTECFErr, nDetTECBErr;
  int nDetTIBTot, nDetTOBTot, nDetTIDFTot, nDetTIDBTot, nDetTECFTot, nDetTECBTot;
  float statusTIB, statusTOB,  statusTIDF,  statusTIDB,  statusTECF,  statusTECB;

  statusTIB  = statusTOB  = statusTIDF  = statusTIDB  = statusTECF  = statusTECB  = -1;
  nDetTIBErr = nDetTOBErr = nDetTIDFErr = nDetTIDBErr = nDetTECFErr = nDetTECBErr = 0;
  nDetTIBTot = nDetTOBTot = nDetTIDFTot = nDetTIDBTot = nDetTECFTot = nDetTECBTot = 0;

  fillDummyGlobalStatus();

  string dname;
  dname = "SiStrip/MechanicalView/TIB";
  fillSubDetStatusFromLayer(dqm_store, dname, nDetTIBTot, nDetTIBErr, 1);
  fillClusterReport(dqm_store, dname, 0);  
  dname = "SiStrip/MechanicalView/TOB";
  fillSubDetStatusFromLayer(dqm_store, dname, nDetTOBTot, nDetTOBErr, 2);  
  fillClusterReport(dqm_store, dname, 4);
  dname = "SiStrip/MechanicalView/TID/side_2";
  fillSubDetStatusFromLayer(dqm_store, dname,  nDetTIDFTot, nDetTIDFErr, 3);  
  fillClusterReport(dqm_store, dname, 10);
  dname = "SiStrip/MechanicalView/TID/side_1";
  fillSubDetStatusFromLayer(dqm_store, dname,  nDetTIDBTot, nDetTIDBErr, 4);  
  fillClusterReport(dqm_store, dname, 13);
  dname = "SiStrip/MechanicalView/TEC/side_2";
  fillSubDetStatusFromLayer(dqm_store, dname,  nDetTECFTot, nDetTECFErr, 5);  
  fillClusterReport(dqm_store, dname, 16);
  dname = "SiStrip/MechanicalView/TEC/side_1";
  fillSubDetStatusFromLayer(dqm_store, dname,  nDetTECBTot, nDetTECBErr, 6);  
  fillClusterReport(dqm_store, dname, 25);
  
  nDetTot = nDetTIBTot + nDetTOBTot + nDetTIDFTot + nDetTIDBTot + nDetTECFTot + nDetTECBTot;
  nDetErr = nDetTIBErr + nDetTOBErr + nDetTIDFErr + nDetTIDBErr + nDetTECFErr + nDetTECBErr;
  if (nDetTot > 0) gStatus = (1 - nDetErr*1.0/nDetTot);
  
  if (nDetTIBTot  > 0) statusTIB  = (1 - nDetTIBErr*1.0/nDetTIBTot);
  if (nDetTOBTot  > 0) statusTOB  = (1 - nDetTOBErr*1.0/nDetTOBTot);
  if (nDetTIDFTot > 0) statusTIDF = (1 - nDetTIDFErr*1.0/nDetTIDFTot);
  if (nDetTIDBTot > 0) statusTIDB = (1 - nDetTIDBErr*1.0/nDetTIDBTot);
  if (nDetTECFTot > 0) statusTECF = (1 - nDetTECFErr*1.0/nDetTECFTot);
  if (nDetTECBTot > 0) statusTECB = (1 - nDetTECBErr*1.0/nDetTECBTot);
  

  SummaryReport->Fill(gStatus);
  
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
}
//
// -- fill subDetStatus
//
void SiStripActionExecutor::fillSubDetStatusFromModule(DQMStore* dqm_store, string& dname,   
  	     int& tot_subdet, int& error_subdet, unsigned int xbin) {
  if (SummaryReportMap->kind() != MonitorElement::DQM_KIND_TH2F) return;
  TH2F* hist2 = SummaryReportMap->getTH2F();
  if (!hist2) return;
  if (dqm_store->dirExists(dname)) {
    dqm_store->cd(dname);
    SiStripFolderOrganizer folder_organizer;
    vector<string> subDirVec = dqm_store->getSubdirs();
    unsigned int ybin = 0;
    tot_subdet = 0, error_subdet =0;
    for (vector<string>::const_iterator ic = subDirVec.begin();
	 ic != subDirVec.end(); ic++) {
      dqm_store->cd((*ic));
      vector<string> mids;
      SiStripUtility::getModuleFolderList(dqm_store, mids);
      int ndet = 0;
      int errdet = 0;       
      for (vector<string>::const_iterator im = mids.begin();
           im != mids.end(); im++) {
         uint32_t detId = atoi((*im).c_str());
	 string subdir_path;
	 folder_organizer.getFolderName(detId, subdir_path);
	 vector<MonitorElement*> meVec = dqm_store->getContents(subdir_path);
         if (meVec.size() == 0) continue;
         ndet++; 
         int err_me = 0;
	 for (vector<MonitorElement*>::const_iterator it = meVec.begin();
	      it != meVec.end(); it++) {
	   MonitorElement * me = (*it);     
	   if (!me) continue;
	   if (me->getQReports().size() == 0) continue;
	   int istat =  SiStripUtility::getMEStatus((*it)); 
	   if (istat == dqm::qstatus::ERROR)   err_me++;
	 }
         if (err_me > 0) errdet++;
      }
      tot_subdet   += ndet;
      error_subdet += errdet;
      ybin++;
      float eff_fac = 1 - (errdet*1.0/ndet);
      if ( ndet > 0) hist2->SetBinContent(xbin,ybin, eff_fac);
    }
  }
}
void SiStripActionExecutor::fillSubDetStatusFromLayer(DQMStore* dqm_store, string& dname,  
	   int& tot_subdet, int& error_subdet, unsigned int xbin) {
  if (SummaryReportMap->kind() != MonitorElement::DQM_KIND_TH2F) return;
  TH2F* hist2 = SummaryReportMap->getTH2F();
  if (!hist2) return;
  if (dqm_store->dirExists(dname)) {
    dqm_store->cd(dname);
    vector<string> subDirVec = dqm_store->getSubdirs();
    unsigned int ybin = 0;
    tot_subdet = error_subdet = 0;
    for (vector<string>::const_iterator ic = subDirVec.begin();
	 ic != subDirVec.end(); ic++) {
      vector<MonitorElement*> meVec;
      meVec = dqm_store->getContents((*ic));
      int errdet = 0;
      int ndet = 0;
      for (vector<MonitorElement*>::const_iterator it = meVec.begin();
               it != meVec.end(); it++) {
	MonitorElement * me = (*it);
	if (!me) continue;
        string name = me->getName();     

	if (me->getQReports().size() != 0 && name.find("Profile") != string::npos) {
	  int nbin = me->getNbinsX();
	  int istat, nbad;
	  istat =  SiStripUtility::getMEStatus((*it), nbad);
	  if (nbin > ndet) ndet = nbin;
	  if (nbad > errdet) errdet = nbad;
        }
      }
      tot_subdet   += ndet;
      error_subdet += errdet;
      ybin++;
      float eff_fac = 1 - (errdet*1.0/ndet);
      if ( ndet > 0) hist2->SetBinContent(xbin,ybin, eff_fac);
    }
  }
}
void SiStripActionExecutor::fillClusterReport(DQMStore* dqm_store, string& dname, int xbin) {
  if (OnTrackClusterReport->kind() != MonitorElement::DQM_KIND_TH1F) return;
  TH1F* hist1 = OnTrackClusterReport->getTH1F();
  if (!hist1) return;
  if (dqm_store->dirExists(dname)) {
    dqm_store->cd(dname);
    vector<string> subDirVec = dqm_store->getSubdirs();
    unsigned int ilayer;
    for (vector<string>::const_iterator ic = subDirVec.begin();
	 ic != subDirVec.end(); ic++) {
      string currDir = (*ic);
      ilayer = atoi((currDir.substr(currDir.find_last_of("_")+1)).c_str());

      vector<MonitorElement*> meVec;
      meVec = dqm_store->getContents(currDir);
      for (vector<MonitorElement*>::const_iterator it = meVec.begin();
	   it != meVec.end(); it++) {
        MonitorElement * me = (*it);
        if (!me) continue;
        string name = me->getName();
        if (name.find("Summary_ClusterStoNCorr__OnTrack") != string::npos) {
          float entries = me->getEntries();
          hist1->SetBinContent(xbin+ilayer, entries);
          break;
        }
      }  
    }
  }
}
//
// -- create reportSummary MEs
//
void SiStripActionExecutor::resetGlobalStatus() {
  if (bookedGlobalStatus_) {
    
    SummaryReport->Reset();
    
    SummaryReportMap->Reset();
    
    SummaryTIB->Reset();
    SummaryTOB->Reset();
    SummaryTIDF->Reset();
    SummaryTIDB->Reset();
    SummaryTECF->Reset();
    SummaryTECB->Reset();

    OnTrackClusterReport->Reset();
  }
}
//
// -- go to a given Directory
//
bool SiStripActionExecutor::goToDir(DQMStore * dqm_store, string name) {
  string currDir = dqm_store->pwd();
  string dirName = currDir.substr(currDir.find_last_of("/")+1);
  if (dirName.find(name) == 0) {
    return true;
  }
  vector<string> subDirVec = dqm_store->getSubdirs();
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    dqm_store->cd(*ic);
    if (!goToDir(dqm_store, name))  dqm_store->goUp();
    else return true;
  }
  return false;  
}
