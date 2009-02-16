#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripMonitorClient/interface/SiStripDcsInfo.h"


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

//
// -- Contructor
//
SiStripDcsInfo::SiStripDcsInfo(edm::ParameterSet const& pSet) {
  // Create MessageSender
  edm::LogInfo( "SiStripDcsInfo") << "SiStripDcsInfo::Deleting SiStripDcsInfo ";

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();
}
SiStripDcsInfo::~SiStripDcsInfo() {
  edm::LogInfo("SiStripDcsInfo") << "SiStripDcsInfo::Deleting SiStripDcsInfo ";

}
//
// -- Begin Job
//
void SiStripDcsInfo::beginJob( const edm::EventSetup &eSetup) {
 

  dqmStore_->setCurrentFolder("SiStrip/EventInfo/DCSContents");

  // Book MEs for SiStrip DAQ fractions
  DcsFraction_= dqmStore_->bookFloat("SiStripDcsFraction");  
  DcsFractionTIB_= dqmStore_->bookFloat("SiStripDcsFraction_TIB");  
  DcsFractionTOB_= dqmStore_->bookFloat("SiStripDcsFraction_TOB");  
  DcsFractionTIDF_= dqmStore_->bookFloat("SiStripDcsFraction_TIDF");  
  DcsFractionTIDB_= dqmStore_->bookFloat("SiStripDcsFraction_TIDB");  
  DcsFractionTECF_= dqmStore_->bookFloat("SiStripDcsFraction_TECF");  
  DcsFractionTECB_= dqmStore_->bookFloat("SiStripDcsFraction_TECB");

  // Fill them with -1 to start with
  DcsFraction_->Fill(-1.0);
  DcsFractionTIB_->Fill(-1.0);
  DcsFractionTOB_->Fill(-1.0);
  DcsFractionTIDF_->Fill(-1.0);
  DcsFractionTIDB_->Fill(-1.0);
  DcsFractionTECF_->Fill(-1.0);
  DcsFractionTECB_->Fill(-1.0);
 
}
//
// -- Begin Run
//
void SiStripDcsInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("SiStripDcsInfo") <<"SiStripDcsInfo:: Begining of Run";

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

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDcsInfo);
