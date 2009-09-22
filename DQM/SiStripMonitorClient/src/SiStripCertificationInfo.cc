#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripCertificationInfo.h"

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

//Run Info
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
using namespace std;
//
// -- Contructor
//
SiStripCertificationInfo::SiStripCertificationInfo(edm::ParameterSet const& pSet) {
  // Create MessageSender
  edm::LogInfo( "SiStripCertificationInfo") << "SiStripCertificationInfo::Deleting SiStripCertificationInfo ";
  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();
  trackingCertificationBooked_ = false;
  sistripCertificationBooked_   = false;
}
SiStripCertificationInfo::~SiStripCertificationInfo() {
  edm::LogInfo("SiStripCertificationInfo") << "SiStripCertificationInfo::Deleting SiStripCertificationInfo ";

}
//
// -- Begin Job
//
void SiStripCertificationInfo::beginJob( const edm::EventSetup &eSetup) {

}
//
// -- Begin Run
//
void SiStripCertificationInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {

  edm::LogInfo ("SiStripCertificationInfo") <<"SiStripCertificationInfo:: Begining of Run";
  unsigned long long cacheID = eSetup.get<SiStripDetCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;       
  }
  eSetup.get<SiStripDetCablingRcd>().get(detCabling_);

  if (!sistripCertificationBooked_)  bookSiStripCertificationMEs();
  if (!trackingCertificationBooked_) bookTrackingCertificationMEs();
}
//
// -- Book MEs for SiStrip Sertification fractions  
//
void SiStripCertificationInfo::bookSiStripCertificationMEs() {
  if (!sistripCertificationBooked_) {
    string strip_dir = "";
    SiStripUtility::getTopFolderPath(dqmStore_, "SiStrip", strip_dir); 
    if (strip_dir.size() > 0) {
      dqmStore_->setCurrentFolder(strip_dir+"/EventInfo");
      SiStripCertification = dqmStore_->bookFloat("CertificationSummary");  
      SubDetMEs local_mes;
      string tag;
      dqmStore_->setCurrentFolder(strip_dir+"/EventInfo/CertificationContents");
      tag = "TIB";
      
      local_mes.folder_name = "TIB";
      local_mes.subdet_tag  = "TIB";
      local_mes.n_layer     = 4;
      local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
      SubDetMEsMap.insert(pair<string, SubDetMEs >(tag, local_mes));
      
      tag = "TOB";
      local_mes.folder_name = "TOB";
      local_mes.subdet_tag  = "TOB";
      local_mes.n_layer     = 6;
      local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
      SubDetMEsMap.insert(pair<string, SubDetMEs >(tag, local_mes));
      
      tag = "TECF";
      local_mes.folder_name = "TEC/side_2";
      local_mes.subdet_tag  = "TEC+";
      local_mes.n_layer     = 9;
      local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
      SubDetMEsMap.insert(pair<string, SubDetMEs >(tag, local_mes));
      
      tag = "TECB";
      local_mes.folder_name = "TEC/side_1";
      local_mes.subdet_tag  = "TEC-";
      local_mes.n_layer     = 9;
      local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
      SubDetMEsMap.insert(pair<string, SubDetMEs >(tag, local_mes));
      
      tag = "TIDF";
      local_mes.folder_name = "TID/side_2";
      local_mes.subdet_tag  = "TID+";
      local_mes.n_layer     = 3;
      local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
      SubDetMEsMap.insert(pair<string, SubDetMEs >(tag, local_mes));
      
      tag = "TIDB";
      local_mes.folder_name = "TID/side_1";
      local_mes.subdet_tag  = "TID-";
      local_mes.n_layer     = 3;
      local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
      SubDetMEsMap.insert(pair<string, SubDetMEs >(tag, local_mes));
      
      dqmStore_->setCurrentFolder(strip_dir+"/EventInfo");
      string  hname  = "CertificationReportMap";
      string  htitle = "SiStrip Certification for Good Detector Fraction";
      SiStripCertificationSummaryMap = dqmStore_->book2D(hname, htitle, 6,0.5,6.5,9,0.5,9.5);
      SiStripCertificationSummaryMap->setAxisTitle("Sub Detector Type", 1);
      SiStripCertificationSummaryMap->setAxisTitle("Layer/Disc Number", 2);
      
      int ibin = 0;
      for (map<string, SubDetMEs>::const_iterator it = SubDetMEsMap.begin(); 
	   it != SubDetMEsMap.end(); it++) {
	ibin++;
	string det = it->first;
	SiStripCertificationSummaryMap->setBinLabel(ibin,det);       
      }
      sistripCertificationBooked_  = true;
      dqmStore_->cd();
    }
  } 
}  
//
// -- Book MEs for SiStrip Sertification fractions  
//
void SiStripCertificationInfo::bookTrackingCertificationMEs() {
  if (!trackingCertificationBooked_) {
    string tracking_dir = "";
    SiStripUtility::getTopFolderPath(dqmStore_, "Tracking", tracking_dir);
    if (tracking_dir.size() > 0) {
      dqmStore_->setCurrentFolder(tracking_dir+"/EventInfo");
      TrackingCertification = dqmStore_->bookFloat("CertificationSummary");  

      dqmStore_->setCurrentFolder(tracking_dir+"/EventInfo/CertificationContents");

      TrackingCertificationRate        = dqmStore_->bookFloat("TrackRate");
      TrackingCertificationChi2overDoF = dqmStore_->bookFloat("TrackChi2overDoF");
      TrackingCertificationRecHits     = dqmStore_->bookFloat("TrackRecHits");
      
      trackingCertificationBooked_ = true;
      dqmStore_->cd();
    }
  }
}
//
// -- Analyze
//
void SiStripCertificationInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) {
}

//
// -- End of Luminosity Block
//
void SiStripCertificationInfo::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup){
  edm::LogInfo ("SiStripCertificationInfo") <<"SiStripCertificationInfo:: Luminosity Block";

  fillDummySiStripCertification();
  fillDummyTrackingCertification();


  float nFEDConnected = 0.0;
  const FEDNumbering numbering;
  const int siStripFedIdMin = numbering.getSiStripFEDIds().first;
  const int siStripFedIdMax = numbering.getSiStripFEDIds().second; 

  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
  if( eSetup.find( recordKey ) != 0) {

    edm::ESHandle<RunInfo> sumFED;
    eSetup.get<RunInfoRcd>().get(sumFED);    
    
    if ( sumFED.isValid() ) {
      vector<int> FedsInIds= sumFED->m_fed_in;   
      for(unsigned int it = 0; it < FedsInIds.size(); ++it) {
	int fedID = FedsInIds[it];     
	if(fedID>=siStripFedIdMin &&  fedID<=siStripFedIdMax)  ++nFEDConnected;
      }
      edm::LogInfo ("SiStripCertificationInfo") <<" SiStripCertificationInfo :: Connected FEDs " << nFEDConnected;
      if (nFEDConnected > 0) {
	fillSiStripCertificationMEs();
	fillTrackingCertificationMEs();
      }
    }
  } 
}
//
// --Fill Tracking Certification 
//
void SiStripCertificationInfo::fillTrackingCertificationMEs() {
  resetTrackingCertificationMEs();
  string tk_dir = "";
  SiStripUtility::getTopFolderPath(dqmStore_, "Tracking", tk_dir);
  if (tk_dir.size() == 0) {
    fillDummyTrackingCertification();
    return;
  }    
  vector<MonitorElement*> all_mes = dqmStore_->getContents(tk_dir+"EventInfo/reportSummaryContents");
  float fval = 1.0;
  for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
      it!= all_mes.end(); it++) {
    MonitorElement * me = (*it);
    if (!me) continue;
    if (me->kind() == MonitorElement::DQM_KIND_REAL) {
      string name = me->getName();
      float val   = me->getFloatValue();
      if (name.find("Rate") != string::npos) TrackingCertificationRate->Fill(val);
      else if (name.find("Chi2overDoF") != string::npos) TrackingCertificationChi2overDoF->Fill(val);
      else if (name.find("RecHits") != string::npos) TrackingCertificationRecHits->Fill(val); 
      fval *= val;
    }
  }  
  TrackingCertification->Fill(fval);  
}
//
// --Fill SiStrip Certification 
//
void SiStripCertificationInfo::fillSiStripCertificationMEs() {
  string mdir = "MechanicalView";
  dqmStore_->cd();
  if (!SiStripUtility::goToDir(dqmStore_, mdir)) return;
  resetSiStripCertificationMEs();
  string mechanical_dir = dqmStore_->pwd();
  uint16_t nDetTot = 0;
  uint16_t nFaultyTot = 0;  
  SiStripFolderOrganizer folder_organizer;  
  int xbin = 0;
  for (map<string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); 
       it != SubDetMEsMap.end(); it++) {   
    xbin++;
    string name = it->first;
    string tag  = it->second.subdet_tag;
    MonitorElement* me = it->second.det_fractionME;
    string bad_module_folder = mechanical_dir+"/"+it->second.folder_name+"/"+"BadModuleList";
    vector<MonitorElement *> faulty_detMEs = dqmStore_->getContents(bad_module_folder);
    
    uint16_t ndet_subdet = 0;
    uint16_t nfaulty_subdet = 0;
    int nlayer = it->second.n_layer;
    int ybin = 0; 
    for (int ilayer = 0; ilayer < nlayer; ilayer++) {
      uint16_t ndet_layer = detCabling_->connectedNumber(tag, ilayer+1);
      ndet_subdet += ndet_layer; 
      ybin++;
      uint16_t nfaulty_layer = 0;
      for (vector<MonitorElement *>::iterator im = faulty_detMEs.begin(); im != faulty_detMEs.end(); im++) {
        if ((*im)->kind() != MonitorElement::DQM_KIND_INT ) continue;
	if ((*im)->getIntValue() == 0) continue; 
        uint32_t detId = atoi((*im)->getName().c_str()); 
        pair<string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detId, false);
        if (abs(det_layer_pair.second) == ilayer+1) nfaulty_layer++; 
      }
      
      nfaulty_subdet += nfaulty_layer;
      float fraction_layer = -1.0;
      if ( ndet_layer > 0) fraction_layer = 1 - ((nfaulty_layer*1.0)/ndet_layer);
      SiStripCertificationSummaryMap->Fill(xbin, ilayer+1,fraction_layer); 
    }
    if (ybin <= SiStripCertificationSummaryMap->getNbinsY()) {
      for (int k = ybin+1; k <= SiStripCertificationSummaryMap->getNbinsY(); k++) SiStripCertificationSummaryMap->Fill(xbin, k, -1.0);    
    }     
    float fraction_subdet = -1.0;
    if (ndet_subdet > 0) fraction_subdet = 1 - ((nfaulty_subdet*1.0)/ndet_subdet);
    if (me) me->Fill(fraction_subdet); 
    nDetTot += ndet_subdet ;
    nFaultyTot += nfaulty_subdet;
  }
  float fraction_global = -1.0;
  if (nDetTot > 0) fraction_global = 1 - ((nFaultyTot *1.8)/nFaultyTot);
  SiStripCertification->Fill(fraction_global);
}
//
// --Reset Tracking Certification 
//
void SiStripCertificationInfo::resetTrackingCertificationMEs() {
  if (!trackingCertificationBooked_) bookTrackingCertificationMEs();
  if (trackingCertificationBooked_) {
    TrackingCertification->Reset(); 
    TrackingCertificationRate->Reset();
    TrackingCertificationChi2overDoF->Reset();
    TrackingCertificationRecHits->Reset();
  }
}
//
// --Fill SiStrip Certification 
//
void SiStripCertificationInfo::resetSiStripCertificationMEs() {
  if (!sistripCertificationBooked_) bookSiStripCertificationMEs();
  if (sistripCertificationBooked_) {
    SiStripCertification->Reset();
    for (map<string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); 
	 it != SubDetMEsMap.end(); it++) {   
      it->second.det_fractionME->Reset();
    }
    SiStripCertificationSummaryMap->Reset();
  }
}
//
// -- Fill Dummy SiStrip Certification
//
void SiStripCertificationInfo::fillDummySiStripCertification() {
  resetSiStripCertificationMEs(); 
  SiStripCertification->Fill(-1.0);
  for (map<string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); 
       it != SubDetMEsMap.end(); it++) {   
    it->second.det_fractionME->Reset();
    it->second.det_fractionME->Fill(-1.0);
  }

  for (int xbin = 1; xbin < SiStripCertificationSummaryMap->getNbinsX()+1; xbin++) {
    for (int ybin = 1; ybin < SiStripCertificationSummaryMap->getNbinsY()+1; ybin++) {
      SiStripCertificationSummaryMap->Fill(xbin, ybin, -1.0);
    }
  }
} 
//
// -- Fill Dummy Tracking Certification 
//
void SiStripCertificationInfo::fillDummyTrackingCertification() {
  resetTrackingCertificationMEs();

  TrackingCertification->Fill(-1.0);

  TrackingCertificationRate->Fill(-1.0);

  TrackingCertificationChi2overDoF->Fill(-1.0);

  TrackingCertificationRecHits->Fill(-1.0);
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripCertificationInfo);
