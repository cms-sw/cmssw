#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripMonitorClient/interface/SiStripDaqInfo.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

//Run Info
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
// FED cabling and numbering
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>
#include <vector>
using namespace std;
//
// -- Contructor
//
SiStripDaqInfo::SiStripDaqInfo(edm::ParameterSet const& pSet) {
  // Create MessageSender
  edm::LogInfo( "SiStripDaqInfo") << "SiStripDaqInfo::Deleting SiStripDaqInfo ";

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();
  nFedTotal = 0;
  bookedStatus_ = false;
}
SiStripDaqInfo::~SiStripDaqInfo() {
  edm::LogInfo("SiStripDaqInfo") << "SiStripDaqInfo::Deleting SiStripDaqInfo ";

}
//
// -- Begin Job
//
void SiStripDaqInfo::beginJob( const edm::EventSetup &eSetup) {
 
}
//
// -- Book MEs for SiStrip Daq Fraction
//
void SiStripDaqInfo::bookStatus() {

  dqmStore_->setCurrentFolder("SiStrip/EventInfo/DAQContents");


  DaqFraction_= dqmStore_->bookFloat("SiStripDaqFraction");  

  for (map<string, vector<unsigned short> >::const_iterator it = subDetFedMap.begin();
       it != subDetFedMap.end(); it++) {
    string det = it->first;

    SubDetMEs local_mes;	
    string me_name;
    me_name = "SiStrip_" + det;    
    local_mes.DaqFractionME = dqmStore_->bookFloat(me_name);  	
    local_mes.ConnectedFeds = 0;
    SubDetMEsMap.insert(make_pair(det, local_mes));
  } 
  bookedStatus_ = true;
  fillDummyStatus();
}

//
// -- Fill with Dummy values
//
void SiStripDaqInfo::fillDummyStatus() {
  if (!bookedStatus_) bookStatus();
  for (map<string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
    it->second.DaqFractionME->Fill(-1.0);
  }
  DaqFraction_->Fill(-1.0);
}
//
// -- Begin Run
//
void SiStripDaqInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("SiStripDaqInfo") <<"SiStripDaqInfo:: Begining of Run";

  // Check latest Fed cabling and create TrackerMapCreator
  unsigned long long cacheID = eSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;       

    edm::ESHandle< SiStripFedCabling > fed_cabling;
    eSetup.get<SiStripFedCablingRcd>().get(fed_cabling); 

    readFedIds(fed_cabling);
  }
  if (!bookedStatus_) bookStatus();  
}
//
// -- Analyze
//
void SiStripDaqInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) {
}

//
// -- Begin Luminosity Block
//
void SiStripDaqInfo::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup){
  edm::LogInfo ("SiStripDaqInfo") <<"SiStripDaqInfo:: Luminosity Block";

  if (nFedTotal == 0) {
   edm::LogInfo ("SiStripDaqInfo") <<" SiStripDaqInfo::No FEDs Connected!!!";
   return;
  }

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
      edm::LogInfo ("SiStripDaqInfo") <<" SiStripDaqInfo::Total # of FEDs " << nFedTotal 
                                      << " Connected FEDs " << nFEDConnected;
      DaqFraction_->Fill(nFEDConnected/nFedTotal);
      readSubdetFedFractions(FedsInIds);
    }
  } 
}
//
// -- Read Sub Detector FEDs
//
void SiStripDaqInfo::readFedIds(const edm::ESHandle<SiStripFedCabling>& fedcabling) {
  const vector<uint16_t>& feds = fedcabling->feds(); 

  nFedTotal = feds.size();
  for(std::vector<unsigned short>::const_iterator ifed = feds.begin(); ifed != feds.end(); ifed++){
    const std::vector<FedChannelConnection> fedChannels = fedcabling->connections( *ifed );
    for (std::vector<FedChannelConnection>::const_iterator iconn = fedChannels.begin(); iconn < fedChannels.end(); iconn++){
      if (!iconn->isConnected()) continue;
      uint32_t detId = iconn->detId();
      if (detId == 0 || detId == 0xFFFFFFFF)  continue;
      std::string subdet_tag;
      SiStripUtility::getSubDetectorTag(detId,subdet_tag);         
      subDetFedMap[subdet_tag].push_back(*ifed); 
      break;
    }
  }
}
//
// -- Fill Subdet FEDIds 
//
void SiStripDaqInfo::readSubdetFedFractions(std::vector<int>& fed_ids) {

  const FEDNumbering numbering;
  const int siStripFedIdMin = numbering.getSiStripFEDIds().first;
  const int siStripFedIdMax = numbering.getSiStripFEDIds().second; 

  // initialiase 
  for (std::map<std::string, std::vector<unsigned short> >::const_iterator it = subDetFedMap.begin();
       it != subDetFedMap.end(); it++) {
    std::string name = it->first;
    map<string, SubDetMEs>::iterator iPos = SubDetMEsMap.find(name);
    if (iPos == SubDetMEsMap.end()) continue;
    iPos->second.ConnectedFeds = 0;
  }
  // count sub detector feds
  for(unsigned int it = 0; it < fed_ids.size(); ++it) {
    unsigned short fedID = fed_ids[it];     
	
    if(fedID>=siStripFedIdMin &&  fedID<=siStripFedIdMax) {

      for (std::map<std::string, std::vector<unsigned short> >::const_iterator it = subDetFedMap.begin();
	   it != subDetFedMap.end(); it++) {
	std::string name = it->first;
	std::vector<unsigned short> subdetIds = it->second;       
        bool fedid_found = false;
	for (std::vector<unsigned short>::iterator iv = subdetIds.begin();
	     iv != subdetIds.end(); iv++) {
	  if ((*iv) == fedID) {
            fedid_found = true;
            break;
          }
        }
        if (fedid_found) {
	  map<string, SubDetMEs>::iterator iPos = SubDetMEsMap.find(name);
	  if (iPos != SubDetMEsMap.end()) iPos->second.ConnectedFeds++;
          break;
	}   
      } 
    }
  }
  
  for (map<string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
    string type = it->first;
    int nFedsConnected   = it->second.ConnectedFeds;
    int nFedsTot = 0;        
    map<std::string,std::vector<unsigned short> >::iterator iPos = subDetFedMap.find(type);
    if (iPos != subDetFedMap.end()) nFedsTot = iPos->second.size();
    if (nFedsTot > 0) it->second.DaqFractionME->Fill(nFedsConnected*1.0/nFedsTot);
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDaqInfo);
