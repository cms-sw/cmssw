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
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

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
//
// -- Contructor
//
SiStripCertificationInfo::SiStripCertificationInfo(edm::ParameterSet const& pSet) :
  m_cacheID_(0) {
  // Create MessageSender
  edm::LogInfo( "SiStripCertificationInfo") << "SiStripCertificationInfo::Deleting SiStripCertificationInfo ";
  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();
  sistripCertificationBooked_   = false;
}
SiStripCertificationInfo::~SiStripCertificationInfo() {
  edm::LogInfo("SiStripCertificationInfo") << "SiStripCertificationInfo::Deleting SiStripCertificationInfo ";

}
//
// -- Begin Job
//
void SiStripCertificationInfo::beginJob() {

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

  nFEDConnected_ = 0;
  const int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID; 

  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
  if( eSetup.find( recordKey ) != 0) {

    edm::ESHandle<RunInfo> sumFED;
    eSetup.get<RunInfoRcd>().get(sumFED);    
    
    if ( sumFED.isValid() ) {
      std::vector<int> FedsInIds= sumFED->m_fed_in;   
      for(unsigned int it = 0; it < FedsInIds.size(); ++it) {
	int fedID = FedsInIds[it];     
	if(fedID>=siStripFedIdMin &&  fedID<=siStripFedIdMax)  ++nFEDConnected_;
      }
      LogDebug ("SiStripDcsInfo") << " SiStripDcsInfo :: Connected FEDs " << nFEDConnected_;
    }
  }
 
  bookSiStripCertificationMEs();
  fillDummySiStripCertification();
  
}
//
// -- Book MEs for SiStrip Sertification fractions  
//
void SiStripCertificationInfo::bookSiStripCertificationMEs() {
  if (!sistripCertificationBooked_) {
    dqmStore_->cd();
    std::string strip_dir = "";
    SiStripUtility::getTopFolderPath(dqmStore_, "SiStrip", strip_dir); 
    if (strip_dir.size() > 0) dqmStore_->setCurrentFolder(strip_dir+"/EventInfo");
    else dqmStore_->setCurrentFolder("SiStrip/EventInfo"); 

    SiStripCertification = dqmStore_->bookFloat("CertificationSummary");  

    std::string  hname  = "CertificationReportMap";
    std::string  htitle = "SiStrip Certification for Good Detector Fraction";
    SiStripCertificationSummaryMap = dqmStore_->book2D(hname, htitle, 6,0.5,6.5,9,0.5,9.5);
    SiStripCertificationSummaryMap->setAxisTitle("Sub Detector Type", 1);
    SiStripCertificationSummaryMap->setAxisTitle("Layer/Disc Number", 2);
    int ibin = 0;
    for (std::map<std::string, SubDetMEs>::const_iterator it = SubDetMEsMap.begin(); 
	 it != SubDetMEsMap.end(); it++) {
      ibin++;
      std::string det = it->first;
      SiStripCertificationSummaryMap->setBinLabel(ibin,det);       
    }

    SubDetMEs local_mes;
    std::string tag;
    dqmStore_->cd();
    if (strip_dir.size() > 0) dqmStore_->setCurrentFolder(strip_dir+"/EventInfo/CertificationContents");
    else dqmStore_->setCurrentFolder("SiStrip/EventInfo/CertificationContents");
    tag = "TIB";
    
    local_mes.folder_name = "TIB";
    local_mes.subdet_tag  = "TIB";
    local_mes.n_layer     = 4;
    local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
    SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));
    
    tag = "TOB";
    local_mes.folder_name = "TOB";
    local_mes.subdet_tag  = "TOB";
    local_mes.n_layer     = 6;
    local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
    SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));
    
    tag = "TECF";
    local_mes.folder_name = "TEC/PLUS";
    local_mes.subdet_tag  = "TEC+";
    local_mes.n_layer     = 9;
    local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
    SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));
    
    tag = "TECB";
    local_mes.folder_name = "TEC/MINUS";
    local_mes.subdet_tag  = "TEC-";
    local_mes.n_layer     = 9;
    local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
    SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));
    
    tag = "TIDF";
    local_mes.folder_name = "TID/PLUS";
    local_mes.subdet_tag  = "TID+";
    local_mes.n_layer     = 3;
    local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
    SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));
    
    tag = "TIDB";
    local_mes.folder_name = "TID/MINUS";
    local_mes.subdet_tag  = "TID-";
    local_mes.n_layer     = 3;
    local_mes.det_fractionME = dqmStore_->bookFloat("SiStrip_"+tag);
    SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));
    
    dqmStore_->cd();
    if (strip_dir.size() > 0) dqmStore_->setCurrentFolder(strip_dir+"/EventInfo");
    
    sistripCertificationBooked_  = true;
    dqmStore_->cd();
  }
}  
//
// -- Analyze
//
void SiStripCertificationInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) {
}
//
// -- End Luminosity Block
//
void SiStripCertificationInfo::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {
  edm::LogInfo( "SiStripDaqInfo") << "SiStripDaqInfo::endLuminosityBlock";

  if (nFEDConnected_ > 0) {
    fillSiStripCertificationMEsAtLumi();  
  }
}

//
// -- End of Run
//
void SiStripCertificationInfo::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
  edm::LogInfo ("SiStripCertificationInfo") <<"SiStripCertificationInfo:: End Run";

  if (nFEDConnected_ > 0) {
    fillSiStripCertificationMEs(eSetup);
  }
}
//
// --Fill SiStrip Certification 
//
void SiStripCertificationInfo::fillSiStripCertificationMEs(edm::EventSetup const& eSetup) {
  if (!sistripCertificationBooked_) {
    edm::LogError("SiStripCertificationInfo") << " SiStripCertificationInfo::fillSiStripCertificationMEs : MEs missing ";
    return;
  }

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  resetSiStripCertificationMEs();
  std::string mdir = "MechanicalView";
  dqmStore_->cd();
  if (!SiStripUtility::goToDir(dqmStore_, mdir)) return;
  std::string mechanical_dir = dqmStore_->pwd();
  uint16_t nDetTot = 0;
  uint16_t nFaultyTot = 0;
  uint16_t nSToNTot = 0; 
  float    sToNTot  = 0.0;
  SiStripFolderOrganizer folder_organizer;  
  int xbin = 0;
  for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); 
       it != SubDetMEsMap.end(); it++) {   
    xbin++;
    std::string name = it->first;
    std::string tag  = it->second.subdet_tag;
    MonitorElement* me = it->second.det_fractionME;
    if (!me) continue;
    std::string bad_module_folder = mechanical_dir+"/"+it->second.folder_name+"/"+"BadModuleList";
    std::vector<MonitorElement *> faulty_detMEs = dqmStore_->getContents(bad_module_folder);
    
    uint16_t ndet_subdet = 0;
    uint16_t nfaulty_subdet = 0;
    int nlayer = it->second.n_layer;
    int ybin = 0; 
    for (int ilayer = 0; ilayer < nlayer; ilayer++) {
      uint16_t ndet_layer = detCabling_->connectedNumber(tag, ilayer+1);
      ndet_subdet += ndet_layer; 
      ybin++;
      uint16_t nfaulty_layer = 0;
      for (std::vector<MonitorElement *>::iterator im = faulty_detMEs.begin(); im != faulty_detMEs.end(); im++) {
        if ((*im)->kind() != MonitorElement::DQM_KIND_INT ) continue;
	if ((*im)->getIntValue() == 0) continue; 
        uint32_t detId = atoi((*im)->getName().c_str()); 
        std::pair<std::string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detId, tTopo, false);
        if (abs(det_layer_pair.second) == ilayer+1) nfaulty_layer++; 
      }
      
      nfaulty_subdet += nfaulty_layer;
      float fraction_layer = -1.0;
      if ( ndet_layer > 0) fraction_layer = 1 - ((nfaulty_layer*1.0)/ndet_layer);
      if (SiStripCertificationSummaryMap) SiStripCertificationSummaryMap->Fill(xbin, ilayer+1,fraction_layer); 
    }
    if (ybin <= SiStripCertificationSummaryMap->getNbinsY()) {
      for (int k = ybin+1; k <= SiStripCertificationSummaryMap->getNbinsY(); k++) SiStripCertificationSummaryMap->Fill(xbin, k, -1.0);    
    }     
    float fraction_subdet = -1.0;
    if (ndet_subdet > 0) fraction_subdet = 1 - ((nfaulty_subdet*1.0)/ndet_subdet);
    // Check S/N status flag and use the minimum between the two
    std::string full_path = mechanical_dir.substr(0, mechanical_dir.find_last_of("/")) 
                            + "/EventInfo/reportSummaryContents/SiStrip_SToNFlag_"+name;
    MonitorElement* me_ston = dqmStore_->get(full_path);
    me->Reset();
    if (me_ston && me_ston->kind()==MonitorElement::DQM_KIND_REAL) {
      float ston_flg = me_ston->getFloatValue(); 
      sToNTot += ston_flg;
      nSToNTot++;
      me->Fill(fminf(fraction_subdet,ston_flg));
    } else me->Fill(fraction_subdet);
    nDetTot += ndet_subdet ;
    nFaultyTot += nfaulty_subdet;
  }
  float fraction_global = -1.0;
  if (nDetTot > 0) fraction_global = 1.0 - ((nFaultyTot*1.0)/nDetTot);
  float ston_frac_global = 1.0;
  if (nSToNTot > 0) ston_frac_global = sToNTot/nSToNTot;   
  SiStripCertification->Fill(fminf(fraction_global,ston_frac_global));
}
//
// --Fill SiStrip Certification 
//
void SiStripCertificationInfo::resetSiStripCertificationMEs() {
  if (!sistripCertificationBooked_) bookSiStripCertificationMEs();
  if (sistripCertificationBooked_) {
    SiStripCertification->Reset();
    for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); 
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
  if (sistripCertificationBooked_) {
    SiStripCertification->Fill(-1.0);
    for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); 
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
} 
//
// --Fill SiStrip Certification
//
void SiStripCertificationInfo::fillSiStripCertificationMEsAtLumi() {
  if (!sistripCertificationBooked_) {
    edm::LogError("SiStripCertificationInfo") << " SiStripCertificationInfo::fillSiStripCertificationMEsAtLumi : MEs missing ";
    return;
  }
  resetSiStripCertificationMEs();
  dqmStore_->cd();
  std::string strip_dir = "";
  SiStripUtility::getTopFolderPath(dqmStore_, "SiStrip", strip_dir);
  if (strip_dir.size() == 0) strip_dir = "SiStrip";

  std::string full_path;
  float dcs_flag = 1.0;
  float dqm_flag = 1.0;
  for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin();
       it != SubDetMEsMap.end(); it++) {
    std::string type = it->first;
    full_path = strip_dir + "/EventInfo/DCSContents/SiStrip_" + type;
    MonitorElement* me_dcs = dqmStore_->get(full_path);
    if (me_dcs && me_dcs->kind() == MonitorElement::DQM_KIND_REAL) dcs_flag = me_dcs->getFloatValue(); 
    full_path = strip_dir + "/EventInfo/reportSummaryContents/SiStrip_" + type;
    MonitorElement* me_dqm = dqmStore_->get(full_path);
    if (me_dqm && me_dqm->kind() == MonitorElement::DQM_KIND_REAL) dqm_flag = me_dqm->getFloatValue(); 
    it->second.det_fractionME->Reset();
    it->second.det_fractionME->Fill(fminf(dqm_flag,dcs_flag));
  }
  dcs_flag = 1.0;
  dqm_flag = 1.0;
  full_path = strip_dir + "/EventInfo/reportSummary";
  MonitorElement* me_dqm = dqmStore_->get(full_path);
  if (me_dqm && me_dqm->kind() == MonitorElement::DQM_KIND_REAL) dqm_flag = me_dqm->getFloatValue();
  full_path = strip_dir + "/EventInfo/DCSSummary";
  MonitorElement* me_dcs = dqmStore_->get(full_path);
  if (me_dcs && me_dcs->kind() == MonitorElement::DQM_KIND_REAL) dcs_flag = me_dcs->getFloatValue();
  SiStripCertification->Reset();
  SiStripCertification->Fill(fminf(dqm_flag,dcs_flag));   
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripCertificationInfo);
