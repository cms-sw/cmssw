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

//
// -- Contructor
//
SiStripDaqInfo::SiStripDaqInfo(edm::ParameterSet const& pSet) {
  // Create MessageSender
  edm::LogInfo( "SiStripDaqInfo") << "SiStripDaqInfo::Deleting SiStripDaqInfo ";

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();
  nFedTotal = 0;
}
SiStripDaqInfo::~SiStripDaqInfo() {
  edm::LogInfo("SiStripDaqInfo") << "SiStripDaqInfo::Deleting SiStripDaqInfo ";

}
//
// -- Begin Job
//
void SiStripDaqInfo::beginJob( const edm::EventSetup &eSetup) {
 

  dqmStore_->setCurrentFolder("SiStrip/EventInfo/DAQContents");

  // Book MEs for SiStrip DAQ fractions
  DaqFraction_= dqmStore_->bookFloat("SiStripDaqFraction");  
  DaqFractionTIB_= dqmStore_->bookFloat("SiStripDaqFraction_TIB");  
  DaqFractionTOB_= dqmStore_->bookFloat("SiStripDaqFraction_TOB");  
  DaqFractionTIDF_= dqmStore_->bookFloat("SiStripDaqFraction_TIDF");  
  DaqFractionTIDB_= dqmStore_->bookFloat("SiStripDaqFraction_TIDB");  
  DaqFractionTECF_= dqmStore_->bookFloat("SiStripDaqFraction_TECF");  
  DaqFractionTECB_= dqmStore_->bookFloat("SiStripDaqFraction_TECB");

  // Fill them with -1 to start with
  DaqFraction_->Fill(-1.0);
  DaqFractionTIB_->Fill(-1.0);
  DaqFractionTOB_->Fill(-1.0);
  DaqFractionTIDF_->Fill(-1.0);
  DaqFractionTIDB_->Fill(-1.0);
  DaqFractionTECF_->Fill(-1.0);
  DaqFractionTECB_->Fill(-1.0);
 
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
      std::vector<int> FedsInIds= sumFED->m_fed_in;   
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
  const std::vector<uint16_t>& feds = fedcabling->feds(); 

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

  float  nFEDConnecteTIB,  nFEDConnecteTOB,  nFEDConnecteTIDF,  nFEDConnecteTIDB,  nFEDConnecteTECF,  nFEDConnecteTECB;
  nFEDConnecteTIB  = nFEDConnecteTOB  = nFEDConnecteTIDF  = nFEDConnecteTIDB  = nFEDConnecteTECF  = nFEDConnecteTECB  = 0;
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
          if (name == "TIB") nFEDConnecteTIB++;
          else if (name == "TOB") nFEDConnecteTOB++;
          else if (name == "TIDF") nFEDConnecteTIDF++;
          else if (name == "TIDB") nFEDConnecteTIDB++;
          else if (name == "TECF") nFEDConnecteTECF++;
          else if (name == "TECB") nFEDConnecteTECB++;
          break;
	}   
      } 
    }
  }
  
  DaqFractionTIB_->Fill(nFEDConnecteTIB/subDetFedMap["TIB"].size());
  DaqFractionTOB_->Fill(nFEDConnecteTOB/subDetFedMap["TOB"].size());
  DaqFractionTIDF_->Fill(nFEDConnecteTIDF/subDetFedMap["TIDF"].size());
  DaqFractionTIDB_->Fill(nFEDConnecteTIDB/subDetFedMap["TIDB"].size());
  DaqFractionTECF_->Fill(nFEDConnecteTECF/subDetFedMap["TECF"].size());
  DaqFractionTECB_->Fill(nFEDConnecteTECB/subDetFedMap["TECB"].size());

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDaqInfo);
