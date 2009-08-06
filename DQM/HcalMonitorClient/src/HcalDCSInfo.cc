#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/HcalMonitorClient/interface/HcalDCSInfo.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

//
// -- Constructor
//
HcalDCSInfo::HcalDCSInfo(edm::ParameterSet const& pSet) {

  debug_ = pSet.getUntrackedParameter<int>("debug",0);

  // Create MessageSender
  edm::LogInfo( "HcalDCSInfo") << "HcalDCSInfo::Creating HcalDCSInfo ";

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();
}

HcalDCSInfo::~HcalDCSInfo() 
{
  edm::LogInfo("HcalDCSInfo") << "HcalDCSInfo::Deleting HcalDCSInfo ";
  
} // destructor

//
// -- Begin Job
//
void HcalDCSInfo::beginJob( const edm::EventSetup &eSetup) {
 

  dqmStore_->setCurrentFolder("Hcal/EventInfo/DCSContents");

  // Book MEs for Hcal DAQ fractions
  DCSFraction_= dqmStore_->bookFloat("Hcal");  
  DCSFractionHB_= dqmStore_->bookFloat("Hcal_HB");  
  DCSFractionHE_= dqmStore_->bookFloat("Hcal_HE");  
  DCSFractionHO_= dqmStore_->bookFloat("Hcal_HO");  
  DCSFractionHF_= dqmStore_->bookFloat("Hcal_HF");  
  DCSFractionZDC_= dqmStore_->bookFloat("Hcal_ZDC");  
  

  // Fill them with -1 to start with
  DCSFraction_->Fill(-1.0);
  DCSFractionHB_->Fill(-1.0);
  DCSFractionHE_->Fill(-1.0);
  DCSFractionHO_->Fill(-1.0);
  DCSFractionHF_->Fill(-1.0);
  DCSFractionZDC_->Fill(-1.0);
} // void HcalDCSInfo::beginJob(...)

//
// -- Begin Run
//
void HcalDCSInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) 
{
  edm::LogInfo ("HcalDCSInfo") <<"HcalDCSInfo:: Begining of Run";
  return;
}

//
// -- Analyze
//
void HcalDCSInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) 
{
}

//
// -- Begin Luminosity Block
//
void HcalDCSInfo::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo ("HcalDCSInfo") <<"HcalDCSInfo:: Luminosity Block";
}

void HcalDCSInfo::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo ("HcalDCSInfo") <<"HcalDCSInfo:: Luminosity Block";
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HcalDCSInfo);
