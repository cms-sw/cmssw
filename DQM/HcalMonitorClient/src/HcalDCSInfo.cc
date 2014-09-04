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
HcalDCSInfo::HcalDCSInfo(const edm::ParameterSet& pSet) {

  debug_ = pSet.getUntrackedParameter<int>("debug",0);
  rootFolder_ = pSet.getUntrackedParameter<std::string>("subSystemFolder","Hcal");
  // Create MessageSender
  edm::LogInfo( "HcalDCSInfo") << "HcalDCSInfo::Creating HcalDCSInfo ";

}

HcalDCSInfo::~HcalDCSInfo() 
{
  edm::LogInfo("HcalDCSInfo") << "HcalDCSInfo::Deleting HcalDCSInfo ";
  
} // destructor

//
// -- bookHistograms
//
void HcalDCSInfo::bookHistograms(DQMStore::IBooker &ib, edm::Run const &run, edm::EventSetup const &es) {
 

  ib.setCurrentFolder(rootFolder_+"/EventInfo/");

  // Book MEs for Hcal DCS fractions

  DCSFraction_= ib.bookFloat("DCSSummary"); 

  DCSSummaryMap_ = ib.book2D("DCSSummaryMap","HcalDCSSummaryMap",7,0.,7.,1,0.,1.);
  DCSSummaryMap_->setAxisRange(-1,1,3);
  DCSSummaryMap_->setBinLabel(1,"HB");
  DCSSummaryMap_->setBinLabel(2,"HE");
  DCSSummaryMap_->setBinLabel(3,"HO");
  DCSSummaryMap_->setBinLabel(4,"HF");
  DCSSummaryMap_->setBinLabel(5,"H00");
  DCSSummaryMap_->setBinLabel(6,"H012");
  DCSSummaryMap_->setBinLabel(7,"HFlumi");
  DCSSummaryMap_->setBinLabel(1,"Status",2);

  ib.setCurrentFolder(rootFolder_+"/EventInfo/DCSContents");
  DCSFractionHB_= ib.bookFloat("Hcal_HB");  
  DCSFractionHE_= ib.bookFloat("Hcal_HE");  
  DCSFractionHO_= ib.bookFloat("Hcal_HO");  
  DCSFractionHF_= ib.bookFloat("Hcal_HF");  
  DCSFractionHO0_= ib.bookFloat("Hcal_HO0");
  DCSFractionHO12_= ib.bookFloat("Hcal_HO12");
  DCSFractionHFlumi_= ib.bookFloat("Hcal_HFlumi");

} 


//
// -- Analyze
//
void HcalDCSInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) 
{
}


/*void HcalDCSInfo::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo ("HcalDCSInfo") <<"HcalDCSInfo:: Luminosity Block";

  // Fill them with -1 to start with

  for (int ii=0;ii<7;ii++) DCSSummaryMap_->setBinContent(ii+1,1,-1.0);
  DCSFraction_->Fill(-1.0);
  DCSFractionHB_->Fill(-1.0);
  DCSFractionHE_->Fill(-1.0);
  DCSFractionHO_->Fill(-1.0);
  DCSFractionHF_->Fill(-1.0);
  DCSFractionHO0_->Fill(-1.0);
  DCSFractionHO12_->Fill(-1.0);
  DCSFractionHFlumi_->Fill(-1.0);

  // Fill them with 1

  for (int ii=0;ii<7;ii++) DCSSummaryMap_->setBinContent(ii+1,1,1.0);
  DCSFraction_->Fill(1.0);
  DCSFractionHB_->Fill(1.0);
  DCSFractionHE_->Fill(1.0);
  DCSFractionHO_->Fill(1.0);
  DCSFractionHF_->Fill(1.0);
  DCSFractionHO0_->Fill(1.0);
  DCSFractionHO12_->Fill(1.0);
  DCSFractionHFlumi_->Fill(1.0);
}*/

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HcalDCSInfo);
