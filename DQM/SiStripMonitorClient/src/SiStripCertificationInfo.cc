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

  if (!sistripCertificationBooked_)  bookSiStripCertificationMEs();
  if (!trackingCertificationBooked_) bookTrackingCertificationMEs();
 
}
//
// -- Book MEs for SiStrip Sertification fractions  
//
void SiStripCertificationInfo::bookSiStripCertificationMEs() {
  dqmStore_->setCurrentFolder("SiStrip/EventInfo");
  SiStripCertification = dqmStore_->bookFloat("CertificationSummary");  

  dqmStore_->setCurrentFolder("SiStrip/EventInfo/CertificationContents");
  SiStripCertificationTIB  = dqmStore_->bookFloat("SiStrip_TIB");  
  SiStripCertificationTOB  = dqmStore_->bookFloat("SiStrip_TOB");  
  SiStripCertificationTIDF = dqmStore_->bookFloat("SiStrip_TIDF");  
  SiStripCertificationTIDB = dqmStore_->bookFloat("SiStrip_TIDB");  
  SiStripCertificationTECF = dqmStore_->bookFloat("SiStrip_TECF");  
  SiStripCertificationTECB = dqmStore_->bookFloat("SiStrip_TECB");

  // Fill them with -1 to start with
  SiStripCertification->Fill(-1.0);
  SiStripCertificationTIB->Fill(-1.0);
  SiStripCertificationTOB->Fill(-1.0);
  SiStripCertificationTIDF->Fill(-1.0);
  SiStripCertificationTIDB->Fill(-1.0);
  SiStripCertificationTECF->Fill(-1.0);
  SiStripCertificationTECB->Fill(-1.0);
  
  sistripCertificationBooked_ = true;
}
//
// -- Book MEs for SiStrip Sertification fractions  
//
void SiStripCertificationInfo::bookTrackingCertificationMEs() {
  dqmStore_->setCurrentFolder("Tracking/EventInfo");
  TrackingCertification = dqmStore_->bookFloat("CertificationSummary");  

  dqmStore_->setCurrentFolder("Tracking/EventInfo/CertificationContents");

  TrackingCertificationRate        = dqmStore_->bookFloat("TrackRate");
  TrackingCertificationChi2overDoF = dqmStore_->bookFloat("TrackChi2overDoF");
  TrackingCertificationRecHits     = dqmStore_->bookFloat("TrackRecHits");


  // Fill them with -1 to start with
  TrackingCertificationRate->Fill(-1.0);
  TrackingCertificationChi2overDoF->Fill(-1.0);
  TrackingCertificationRecHits->Fill(-1.0);
 
  TrackingCertification->Fill(-1.0);

  trackingCertificationBooked_ = true;
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
  fillTrackingCertificationMEs();
  //  fillSiStripCertificationMEs();
}
//
// --Fill Tracking Certification 
//
void SiStripCertificationInfo::fillTrackingCertificationMEs() {
  string tkreport_dir = "Tracking/EventInfo/reportSummaryContents";
  if (!dqmStore_->dirExists(tkreport_dir)) return;
  resetTrackingCertificationMEs();
  vector<MonitorElement*> all_mes = dqmStore_->getContents(tkreport_dir);
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
}
//
// --Reset Tracking Certification 
//
void SiStripCertificationInfo::resetTrackingCertificationMEs() {
  if (!trackingCertificationBooked_) bookTrackingCertificationMEs();
  TrackingCertification->Reset();    
}
//
// --Fill SiStrip Certification 
//
void SiStripCertificationInfo::resetSiStripCertificationMEs() {
  if (!sistripCertificationBooked_) bookSiStripCertificationMEs();
  SiStripCertification->Reset();
  SiStripCertificationTIB->Reset();
  SiStripCertificationTOB->Reset();
  SiStripCertificationTIDF->Reset();
  SiStripCertificationTIDB->Reset();
  SiStripCertificationTECF->Reset();
  SiStripCertificationTECB->Reset();
   
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripCertificationInfo);
