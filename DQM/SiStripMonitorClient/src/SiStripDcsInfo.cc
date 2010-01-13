#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripDcsInfo.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

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
SiStripDcsInfo::SiStripDcsInfo(edm::ParameterSet const& pSet) {
  // Create MessageSender
  edm::LogInfo( "SiStripDcsInfo") << "SiStripDcsInfo::Deleting SiStripDcsInfo ";

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();
  bookedStatus_ = false;
}
//
// -- Destructor
//
SiStripDcsInfo::~SiStripDcsInfo() {
  edm::LogInfo("SiStripDcsInfo") << "SiStripDcsInfo::Deleting SiStripDcsInfo ";

}
//
// -- Begin Job
//
void SiStripDcsInfo::beginJob() {
 
}
//
// -- Book MEs for SiStrip Dcs Fraction
//
void SiStripDcsInfo::bookStatus() {

  if (!bookedStatus_) {
    string strip_dir = "";
    SiStripUtility::getTopFolderPath(dqmStore_, "SiStrip", strip_dir); 
    if (strip_dir.size() > 0) {
      dqmStore_->setCurrentFolder(strip_dir+"/EventInfo");
      
      
      DcsFraction_= dqmStore_->bookFloat("DCSSummary");  
      
      dqmStore_->setCurrentFolder(strip_dir+"/EventInfo/DCSContents");
      vector<string> det_type;
      det_type.push_back("TIB");
      det_type.push_back("TOB");
      det_type.push_back("TIDF");
      det_type.push_back("TIDB");
      det_type.push_back("TECF");
      det_type.push_back("TECB");
      
      for ( vector<string>::iterator it = det_type.begin(); it != det_type.end(); it++) {
	SubDetMEs local_mes;	
	string me_name;
	string det = (*it);
	me_name = "SiStrip_" + det;    
	local_mes.DcsFractionME = dqmStore_->bookFloat(me_name);  	
	local_mes.TotalDetectors = 0;
	local_mes.FaultyDetectors = 0;
	SubDetMEsMap.insert(std::make_pair(det, local_mes));
      } 
      bookedStatus_ = true;
      dqmStore_->cd();
    }
  }
}
//
// -- Fill with Dummy values
//
void SiStripDcsInfo::fillDummyStatus() {
  if (!bookedStatus_) bookStatus();
  if (bookedStatus_) {
    for (map<string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
      it->second.DcsFractionME->Reset();
      it->second.DcsFractionME->Fill(-1.0);
    }
    DcsFraction_->Reset();
    DcsFraction_->Fill(-1.0);
  }
}
//
// -- Begin Run
//
void SiStripDcsInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("SiStripDcsInfo") <<"SiStripDcsInfo:: Begining of Run";
  unsigned long long cacheID = eSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;
    edm::LogInfo("SiStripDcsInfo") <<"SiStripDcsInfo::beginRun: "
                                    << " Change in Cache";
    eSetup.get<SiStripDetVOffRcd>().get(siStripDetVOff_);
    eSetup.get<SiStripDetCablingRcd>().get(detCabling_);
  }
  if (!bookedStatus_) bookStatus();

  nFEDConnected = 0;
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
      edm::LogInfo ("SiStripDcsInfo") << " SiStripDcsInfo :: Connected FEDs " << nFEDConnected;
    }
  } 
}
//
// -- Analyze
//
void SiStripDcsInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) {
}
//
// -- End Luminosity Block
//
void SiStripDcsInfo::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {
  edm::LogInfo( "SiStripDcsInfo") << "SiStripDcsInfo::endLuminosityBlock";
}
//
// -- End Run
//
void SiStripDcsInfo::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
  edm::LogInfo ("SiStripDcsInfo") <<"SiStripDcsInfo::EndRun";

  fillDummyStatus();

  if (nFEDConnected > 0) fillStatus();

}
//
// -- Get Faulty Detectors
//
void SiStripDcsInfo::readStatus() {
  
  std::vector<uint32_t> SelectedDetIds;
  detCabling_->addActiveDetectorsRawIds(SelectedDetIds);
  edm::LogInfo( "SiStripDcsInfo") << " SiStripDcsInfo::readStatus : "  
                                  << " Total Detectors " << SelectedDetIds.size();
 
  std::vector <uint32_t> FaultyDetIds;
  siStripDetVOff_->getDetIds(FaultyDetIds);
  edm::LogInfo( "SiStripDcsInfo") << " SiStripDcsInfo::readStatus : "
                                  << " Faulty Detectors " << FaultyDetIds.size();

  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(); idetid != SelectedDetIds.end(); ++idetid){    
    uint32_t detId = *idetid;
    bool hv_error = false;
    if (detId == 0 || detId == 0xFFFFFFFF) continue;

    for (std::vector<uint32_t>::const_iterator ihvoff=FaultyDetIds.begin(); ihvoff!=FaultyDetIds.end();++ihvoff){
      uint32_t detId_hvoff = (*ihvoff);
      if (detId_hvoff == detId) {
	hv_error = true;
        break;
      }
    }
    string subdet_tag;
    SiStripUtility::getSubDetectorTag(detId,subdet_tag);         

    map<string, SubDetMEs>::iterator iPos = SubDetMEsMap.find(subdet_tag);
    if (iPos != SubDetMEsMap.end()){    
      iPos->second.TotalDetectors++;
      if (hv_error) {
	iPos->second.FaultyDetectors++;
        addBadModules(detId);
      }
    }
  }
}
//
// -- 
//
void SiStripDcsInfo::fillStatus(){
  
  readStatus();
  if (!bookedStatus_) bookStatus();
  for (map<string,SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
    int total_det  = it->second.TotalDetectors;
    int faulty_det = it->second.FaultyDetectors; 
    if  (total_det > 0) {
      float fraction = 1.0  - faulty_det*1.0/total_det;   
      it->second.DcsFractionME->Reset();
      it->second.DcsFractionME->Fill(fraction);
      edm::LogInfo( "SiStripDcsInfo") << " SiStripDcsInfo::fillStatus : Sub Detector "
                  << it->first << "  " << total_det  << " " << faulty_det << endl;
    }
  } 
}
//
// -- Add Bad Modules
//
void SiStripDcsInfo::addBadModules(uint32_t det_id) {
  
  dqmStore_->cd();
  string mdir = "MechanicalView";
  if (!SiStripUtility::goToDir(dqmStore_, mdir)) return;
  string mechanical_dir = dqmStore_->pwd();
  string tag = "DCSError";
  string subdet_folder ;
  SiStripFolderOrganizer folder_organizer;
  folder_organizer.getSubDetFolder(det_id,subdet_folder);
  if (!dqmStore_->dirExists(subdet_folder)) {
    subdet_folder = mechanical_dir + subdet_folder.substr(subdet_folder.find("MechanicalView")+14);
    if (!dqmStore_->dirExists(subdet_folder)) return;
  }
  string bad_module_folder = subdet_folder + "/" + "BadModuleList";
  dqmStore_->setCurrentFolder(bad_module_folder);

  ostringstream detid_str;
  detid_str << det_id;
  string full_path = bad_module_folder + "/" + detid_str.str();
  MonitorElement* me = dqmStore_->get(full_path);
  uint16_t flag = 0; 
  if (me) {
    flag = me->getIntValue();
    me->Reset();
  } else me = dqmStore_->bookInt(detid_str.str());
  SiStripUtility::setBadModuleFlag(tag, flag);
  me->Fill(flag);
  
  dqmStore_->cd();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDcsInfo);
