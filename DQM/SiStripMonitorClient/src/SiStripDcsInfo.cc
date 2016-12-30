#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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

//
// -- Contructor
//
SiStripDcsInfo::SiStripDcsInfo(edm::ParameterSet const& pSet) : 
    m_cacheIDCabling_(0),
    m_cacheIDDcs_(0),
    bookedStatus_(false)
{ 
  // Create MessageSender
  LogDebug( "SiStripDcsInfo") << "SiStripDcsInfo::Deleting SiStripDcsInfo ";

  std::string tag;
  SubDetMEs local_mes;
  
  tag = "TIB";   
  local_mes.folder_name = "TIB";
  local_mes.DcsFractionME  = 0;
  local_mes.TotalDetectors = 0; 
  local_mes.FaultyDetectors.clear();
  SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));
      
  tag = "TOB";
  local_mes.folder_name = "TOB";
  local_mes.DcsFractionME  = 0;
  local_mes.TotalDetectors = 0; 
  local_mes.FaultyDetectors.clear();
  SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));

  tag = "TECB";
  local_mes.folder_name = "TEC/MINUS";
  local_mes.DcsFractionME  = 0;
  local_mes.TotalDetectors = 0; 
  local_mes.FaultyDetectors.clear();
  SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));

  tag = "TECF";
  local_mes.folder_name = "TEC/PLUS";
  local_mes.DcsFractionME  = 0;
  local_mes.TotalDetectors = 0; 
  local_mes.FaultyDetectors.clear();
  SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));

  tag = "TIDB";
  local_mes.folder_name = "TID/MINUS";
  local_mes.DcsFractionME  = 0;
  local_mes.TotalDetectors = 0; 
  local_mes.FaultyDetectors.clear();
  SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));

  tag = "TIDF";
  local_mes.folder_name = "TID/PLUS";
  local_mes.DcsFractionME  = 0;
  local_mes.TotalDetectors = 0; 
  local_mes.FaultyDetectors.clear();
  SubDetMEsMap.insert(std::pair<std::string, SubDetMEs >(tag, local_mes));
}
//
// -- Destructor
//
SiStripDcsInfo::~SiStripDcsInfo() {
  LogDebug("SiStripDcsInfo") << "SiStripDcsInfo::Deleting SiStripDcsInfo ";

}
//
// -- Begin Run
//
void SiStripDcsInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  tTopo = tTopoHandle.product();

  LogDebug ("SiStripDcsInfo") <<"SiStripDcsInfo:: Begining of Run";
  nFEDConnected_ = 0;
  const int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID; 

  // Count Tracker FEDs from RunInfo
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

  if (nFEDConnected_ > 0) readCabling(eSetup);
}
//
// -- Begin Luminosity Block
//
void SiStripDcsInfo::dqmBeginLuminosityBlock(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  LogDebug( "SiStripDcsInfo") << "SiStripDcsInfo::beginLuminosityBlock";
  
  if (nFEDConnected_ == 0) return;

  if (!bookedStatus_) 
    {
      bookStatus(ibooker , igetter);
      fillDummyStatus(ibooker , igetter);
    }

  // initialise BadModule list 
  for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
    it->second.FaultyDetectors.clear();
  }
  readStatus(eSetup);
}

//
// -- End Luminosity Block
//
void SiStripDcsInfo::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  LogDebug( "SiStripDcsInfo") << "SiStripDcsInfo::endLuminosityBlock";

  if (nFEDConnected_ == 0) return;
  readStatus(eSetup);
  fillStatus(ibooker , igetter);
}
//
// -- End Run
//
void SiStripDcsInfo::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
  LogDebug ("SiStripDcsInfo") <<"SiStripDcsInfo::EndRun";

  if (nFEDConnected_ == 0) return;

  for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
    it->second.FaultyDetectors.clear();
  }
  readStatus(eSetup); 
} 

void SiStripDcsInfo::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter){

  if (nFEDConnected_ == 0) 
    fillDummyStatus(ibooker , igetter);
  
  addBadModules(ibooker , igetter);
  fillStatus(ibooker , igetter);
}

//
// -- Book MEs for SiStrip Dcs Fraction
//
void SiStripDcsInfo::bookStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (!bookedStatus_) {
    std::string strip_dir = "";
    SiStripUtility::getTopFolderPath(ibooker , igetter , "SiStrip", strip_dir); 
    if (strip_dir.size() > 0) ibooker.setCurrentFolder(strip_dir+"/EventInfo");
    else ibooker.setCurrentFolder("SiStrip/EventInfo");
       
    DcsFraction_= ibooker.bookFloat("DCSSummary");  
    
    DcsFraction_->setLumiFlag();
    
    ibooker.cd();
    if (strip_dir.size() > 0)  ibooker.setCurrentFolder(strip_dir+"/EventInfo/DCSContents");
    else ibooker.setCurrentFolder("SiStrip/EventInfo/DCSContents"); 
    for (std::map<std::string,SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
      SubDetMEs local_mes;	
      std::string me_name;
      me_name = "SiStrip_" + it->first;
      it->second.DcsFractionME = ibooker.bookFloat(me_name);  
      it->second.DcsFractionME->setLumiFlag();	
    } 
    bookedStatus_ = true;
    ibooker.cd();
  }
}
//
// -- Read Cabling
// 
void SiStripDcsInfo::readCabling(edm::EventSetup const& eSetup) {

  unsigned long long cacheID = eSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (m_cacheIDCabling_ != cacheID) {
    m_cacheIDCabling_ = cacheID;
    LogDebug("SiStripDcsInfo") <<"SiStripDcsInfo::readCabling : "
				   << " Change in Cache";
    eSetup.get<SiStripDetCablingRcd>().get(detCabling_);

    std::vector<uint32_t> SelectedDetIds;
    detCabling_->addActiveDetectorsRawIds(SelectedDetIds);
    LogDebug( "SiStripDcsInfo") << " SiStripDcsInfo::readCabling : "  
				    << " Total Detectors " << SelectedDetIds.size();
    

    // initialise total # of detectors first
    for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
      it->second.TotalDetectors = 0;
    }
    
    for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(); idetid != SelectedDetIds.end(); ++idetid){    
      uint32_t detId = *idetid;
      if (detId == 0 || detId == 0xFFFFFFFF) continue;
      std::string subdet_tag;
      SiStripUtility::getSubDetectorTag(detId,subdet_tag,tTopo);         
      
      std::map<std::string, SubDetMEs>::iterator iPos = SubDetMEsMap.find(subdet_tag);
      if (iPos != SubDetMEsMap.end()){    
	iPos->second.TotalDetectors++;
      }
    }
  }
}
//
// -- Get Faulty Detectors
//
void SiStripDcsInfo::readStatus(edm::EventSetup const& eSetup) {

  eSetup.get<SiStripDetVOffRcd>().get(siStripDetVOff_);
  std::vector <uint32_t> FaultyDetIds;
  siStripDetVOff_->getDetIds(FaultyDetIds);
  LogDebug( "SiStripDcsInfo") << " SiStripDcsInfo::readStatus : "
				  << " Faulty Detectors " << FaultyDetIds.size();
  // Read and fille bad modules
  for (std::vector<uint32_t>::const_iterator ihvoff=FaultyDetIds.begin(); ihvoff!=FaultyDetIds.end();++ihvoff){
    uint32_t detId_hvoff = (*ihvoff);
    if (!detCabling_->IsConnected(detId_hvoff)) continue;    
    std::string subdet_tag;
    SiStripUtility::getSubDetectorTag(detId_hvoff,subdet_tag,tTopo);         
    
    std::map<std::string, SubDetMEs>::iterator iPos = SubDetMEsMap.find(subdet_tag);
    if (iPos != SubDetMEsMap.end()){  
      std::vector<uint32_t>::iterator ibad = std::find(iPos->second.FaultyDetectors.begin(), iPos->second.FaultyDetectors.end(), detId_hvoff);
      if (ibad ==  iPos->second.FaultyDetectors.end()) iPos->second.FaultyDetectors.push_back( detId_hvoff);
    }
  }
}
//
// -- Fill Status
//
void SiStripDcsInfo::fillStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter){
  if (!bookedStatus_) bookStatus(ibooker , igetter);
  if (bookedStatus_) {
    float total_det = 0.0;
    float faulty_det = 0.0;
    float fraction;
    for (std::map<std::string,SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
      int total_subdet  = it->second.TotalDetectors;
      int faulty_subdet = it->second.FaultyDetectors.size(); 
      if  (nFEDConnected_ == 0  || total_subdet == 0) fraction = -1;
      else fraction = 1.0  - faulty_subdet*1.0/total_subdet;   
      it->second.DcsFractionME->Reset();
      it->second.DcsFractionME->Fill(fraction);
      edm::LogInfo( "SiStripDcsInfo") << " SiStripDcsInfo::fillStatus : Sub Detector "
	<< it->first << " Total Number " << total_subdet  
        << " Faulty ones " << faulty_subdet;
      total_det += total_subdet;
      faulty_det += faulty_subdet;
    }
    if (nFEDConnected_ == 0 || total_det == 0) fraction = -1.0;
    else fraction = 1 - faulty_det/total_det;
    DcsFraction_->Reset();
    DcsFraction_->Fill(fraction);
  } 
}
//
// -- Fill with Dummy values
//
void SiStripDcsInfo::fillDummyStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (!bookedStatus_) bookStatus(ibooker , igetter);
  if (bookedStatus_) {
    for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
      it->second.DcsFractionME->Reset();
      it->second.DcsFractionME->Fill(-1.0);
    }
    DcsFraction_->Reset();
    DcsFraction_->Fill(-1.0);
  }
}
//
// -- Add Bad Modules
//
void SiStripDcsInfo::addBadModules(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
    
  ibooker.cd();
  std::string mdir = "MechanicalView";
  if (!SiStripUtility::goToDir(ibooker , igetter , mdir)) {
    ibooker.setCurrentFolder("SiStrip/"+mdir);
  }
  std::string mechanical_dir = ibooker.pwd();
  std::string tag = "DCSError";

  for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
    std::vector<uint32_t> badModules = it->second.FaultyDetectors; 
    for (std::vector<uint32_t>::iterator ibad = badModules.begin(); 
	 ibad != badModules.end(); ibad++) {

      std::string bad_module_folder = mechanical_dir + "/" +
                                      it->second.folder_name + "/"     
                                      "BadModuleList";      
      ibooker.setCurrentFolder(bad_module_folder);

      std::ostringstream detid_str;
      detid_str << (*ibad);
      std::string full_path = bad_module_folder + "/" + detid_str.str();
      MonitorElement* me = igetter.get(full_path);
      uint16_t flag = 0; 
      if (me) {
	flag = me->getIntValue();
	me->Reset();
      } else me = ibooker.bookInt(detid_str.str());
      SiStripUtility::setBadModuleFlag(tag, flag);
      me->Fill(flag);
    }
  }   
  ibooker.cd();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDcsInfo);
