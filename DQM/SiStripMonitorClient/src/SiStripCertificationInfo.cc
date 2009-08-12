#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripMonitorClient/interface/SiStripCertificationInfo.h"


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
SiStripCertificationInfo::SiStripCertificationInfo(edm::ParameterSet const& pSet) {
  // Create MessageSender
  edm::LogInfo( "SiStripCertificationInfo") << "SiStripCertificationInfo::Deleting SiStripCertificationInfo ";

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();
}
SiStripCertificationInfo::~SiStripCertificationInfo() {
  edm::LogInfo("SiStripCertificationInfo") << "SiStripCertificationInfo::Deleting SiStripCertificationInfo ";

}
//
// -- Begin Job
//
void SiStripCertificationInfo::beginJob( const edm::EventSetup &eSetup) {
 

  // Book MEs for SiStrip DAQ fractions  
  dqmStore_->setCurrentFolder("SiStrip/EventInfo");
  CertificationBit_= dqmStore_->bookFloat("SiStripCertificationFraction");  

  dqmStore_->setCurrentFolder("SiStrip/EventInfo/CertificationContents");
  CertificationBitTIB_ = dqmStore_->bookFloat("SiStrip_TIB");  
  CertificationBitTOB_ = dqmStore_->bookFloat("SiStrip_TOB");  
  CertificationBitTIDF_= dqmStore_->bookFloat("SiStrip_TIDF");  
  CertificationBitTIDB_= dqmStore_->bookFloat("SiStrip_TIDB");  
  CertificationBitTECF_= dqmStore_->bookFloat("SiStrip_TECF");  
  CertificationBitTECB_= dqmStore_->bookFloat("SiStrip_TECB");

  // Fill them with -1 to start with
  CertificationBit_->Fill(-1.0);
  CertificationBitTIB_->Fill(-1.0);
  CertificationBitTOB_->Fill(-1.0);
  CertificationBitTIDF_->Fill(-1.0);
  CertificationBitTIDB_->Fill(-1.0);
  CertificationBitTECF_->Fill(-1.0);
  CertificationBitTECB_->Fill(-1.0);
}
//
// -- Begin Run
//
void SiStripCertificationInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("SiStripCertificationInfo") <<"SiStripCertificationInfo:: Begining of Run";

}
//
// -- Analyze
//
void SiStripCertificationInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) {
}

//
// -- Begin Luminosity Block
//
void SiStripCertificationInfo::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup){
  edm::LogInfo ("SiStripCertificationInfo") <<"SiStripCertificationInfo:: Luminosity Block";

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripCertificationInfo);
