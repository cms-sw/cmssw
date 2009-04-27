#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripMonitorClient/interface/SiStripDcsInfo.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

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
void SiStripDcsInfo::beginJob( const edm::EventSetup &eSetup) {
 
  if (!bookedStatus_) bookStatus();
}
//
// -- Book MEs for SiStrip Dcs Fraction
//
void SiStripDcsInfo::bookStatus() {

  dqmStore_->setCurrentFolder("SiStrip/EventInfo/DCSContents");


  DcsFraction_= dqmStore_->bookFloat("SiStripDcsFraction");  
 
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
    me_name = "SiStripDcsFraction_" + det;    
    local_mes.DcsFractionME = dqmStore_->bookFloat(me_name);  	
    local_mes.TotalDetectors = 0;
    local_mes.FaultyDetectors = 0;
    SubDetMEsMap.insert(std::make_pair(det, local_mes));
  } 
  bookedStatus_ = true;
  fillDummyStatus();
}
//
// -- Fill with Dummy values
//
void SiStripDcsInfo::fillDummyStatus() {
  if (!bookedStatus_) bookStatus();
  for (map<string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
    it->second.DcsFractionME->Fill(-1.0);
  }
  DcsFraction_->Fill(-1.0);
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
}
//
// -- Analyze
//
void SiStripDcsInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) {
}

//
// -- Begin Luminosity Block
//
void SiStripDcsInfo::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup){
  edm::LogInfo ("SiStripDcsInfo") <<"SiStripDcsInfo:: Luminosity Block";

  if (!bookedStatus_) bookStatus();
  fillStatus();
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

  string subdet_tag;
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
    StripSubdetector subdet(*idetid);
    
    switch (subdet.subdetId()) 
      {
      case StripSubdetector::TIB:
	{
          subdet_tag = "TIB";
          break;
	}
      case StripSubdetector::TID:
	{
	  TIDDetId tidId(detId);
	  if (tidId.side() == 2) {
            subdet_tag = "TIDF";
	  }  else if (tidId.side() == 1) {
	    subdet_tag = "TIDB";
	  }
	  break;       
	}
      case StripSubdetector::TOB:
	{
          subdet_tag = "TOB";
          break;
	}
      case StripSubdetector::TEC:
	{
	  TECDetId tecId(detId);
	  if (tecId.side() == 2) {
	    subdet_tag = "TECF";
	  }  else if (tecId.side() == 1) {
            subdet_tag = "TECB";	
	  }
	  break;       
	}
      }
    map<string, SubDetMEs>::iterator iPos = SubDetMEsMap.find(subdet_tag);
    if (iPos != SubDetMEsMap.end()){    
      iPos->second.TotalDetectors++;
      if (hv_error)  iPos->second.FaultyDetectors++;
    }
  }
}
//
// -- 
//
void SiStripDcsInfo::fillStatus(){
  
  readStatus();
  for (map<string,SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
    int total_det  = it->second.TotalDetectors;
    int faulty_det = it->second.FaultyDetectors; 
    if  (total_det > 0) {
      float fraction = 1.0  - faulty_det*1.0/total_det;   
      it->second.DcsFractionME->Fill(fraction);
      edm::LogInfo( "SiStripDcsInfo") << " SiStripDcsInfo::fillStatus : Sub Detector " << it->first << "  " 
				      << total_det  << " " << faulty_det << endl;
    }
  } 
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDcsInfo);
