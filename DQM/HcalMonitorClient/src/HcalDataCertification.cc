// -*- C++ -*-
//
// Package:    DQMO/HcalMonitorClient/HcalDataCertification
// Class:      HcalDataCertification
// 
/**\class HcalDataCertification HcalDataCertification.cc DQM/HcalMonitorClient/src/HcalDataCertification.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Igor Vodopiyanov"
//         Created:  Nov-21 2008
// $Id: HcalDataCertification.cc,v 1.13 2010/08/12 21:09:45 temple Exp $
//
//

// system include files
#include <memory>
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <exception>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class HcalDataCertification : public edm::EDAnalyzer {
   public:
      explicit HcalDataCertification(const edm::ParameterSet&);
      ~HcalDataCertification();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) ;
      virtual void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) ;
      void endRun(const edm::Run & r, const edm::EventSetup & c);
  void CertifyHcal();

   // ----------member data ---------------------------

   edm::ParameterSet conf_;
   DQMStore * dbe_;
   MonitorElement* CertificationSummary;
   MonitorElement* CertificationSummaryMap;
   MonitorElement* Hcal_HB;
   MonitorElement* Hcal_HE;
   MonitorElement* Hcal_HF;
   MonitorElement* Hcal_HO;
   MonitorElement* Hcal_HFlumi;
   MonitorElement* Hcal_HO0;
   MonitorElement* Hcal_HO12;
   int debug_;
   std::string rootFolder_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

HcalDataCertification::HcalDataCertification(const edm::ParameterSet& iConfig)
{
  // now do what ever initialization is needed
  debug_ = iConfig.getUntrackedParameter<int>("debug",0);
  rootFolder_ = iConfig.getUntrackedParameter<std::string>("subSystemFolder","Hcal");
  dbe_ = edm::Service<DQMStore>().operator->();  
}

HcalDataCertification::~HcalDataCertification()
{ 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void
HcalDataCertification::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalDataCertification::beginJob()
{
  if (debug_>0) std::cout<<"<HcalDataCertification> beginJob"<< std::endl;

  dbe_->setCurrentFolder(rootFolder_);
  std::string currDir = dbe_->pwd();
  if (debug_>0) std::cout << "--- Current Directory " << currDir << std::endl;
  std::vector<MonitorElement*> mes = dbe_->getAllContents("");
  if (debug_>0) std::cout << "found " << mes.size() << " monitoring elements:" << std::endl;

  dbe_->setCurrentFolder(rootFolder_+"/EventInfo/");

  CertificationSummary = dbe_->bookFloat("CertificationSummary");

  CertificationSummaryMap = dbe_->book2D("CertificationSummaryMap","HcalCertificationSummaryMap",7,0.,7.,1,0.,1.);
  CertificationSummaryMap->setAxisRange(-1,1,3);
  CertificationSummaryMap->setBinLabel(1,"HB");
  CertificationSummaryMap->setBinLabel(2,"HE");
  CertificationSummaryMap->setBinLabel(3,"HO");
  CertificationSummaryMap->setBinLabel(4,"HF");
  CertificationSummaryMap->setBinLabel(5,"H00");
  CertificationSummaryMap->setBinLabel(6,"H012");
  CertificationSummaryMap->setBinLabel(7,"HFlumi");
  CertificationSummaryMap->setBinLabel(1,"Status",2);

  dbe_->setCurrentFolder(rootFolder_+"/EventInfo/CertificationContents/");
  Hcal_HB = dbe_->bookFloat("Hcal_HB");
  Hcal_HE = dbe_->bookFloat("Hcal_HE");
  Hcal_HF = dbe_->bookFloat("Hcal_HF");
  Hcal_HO = dbe_->bookFloat("Hcal_HO");
  Hcal_HFlumi = dbe_->bookFloat("Hcal_HFlumi");
  Hcal_HO0    = dbe_->bookFloat("Hcal_HO0");
  Hcal_HO12   = dbe_->bookFloat("Hcal_HO12");

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalDataCertification::endJob() 
{
  if (debug_>0) std::cout << "<HcalDataCertification> endJob " << std::endl;
}

// ------------ method called just before starting a new run  ------------
void 
HcalDataCertification::beginLuminosityBlock(const edm::LuminosityBlock& run, const edm::EventSetup& c)
{
  if (debug_>0) std::cout<<"<HcalDataCertification> beginLuminosityBlock"<<std::endl;
}

// ------------ method called right after a run ends ------------
void 
HcalDataCertification::endLuminosityBlock(const edm::LuminosityBlock& run, const edm::EventSetup& c)
{
  CertifyHcal();
}

void HcalDataCertification::endRun(const edm::Run & r, const edm::EventSetup & c)
{
  CertifyHcal();
}

void HcalDataCertification::CertifyHcal()
{

  float hcalFrac,reportFrac,dcsFrac,daqFrac;
  float fracHCAL[7][3];
  float certHcal[7];

  if (debug_>0) {
    dbe_->setCurrentFolder(rootFolder_);
    std::string currDir = dbe_->pwd();
    std::cout << "<HcalDataCertification::endLuminosityBlock> --- Current Directory " << currDir << std::endl;
    std::vector<MonitorElement*> mes = dbe_->getAllContents("");
    std::cout << "found " << mes.size() << " monitoring elements:" << std::endl;
  }

  if (dbe_->get(rootFolder_+"/EventInfo/DCSSummary")) {
    dcsFrac = (dbe_->get(rootFolder_+"/EventInfo/DCSSummary"))->getFloatValue();
  }
  else dcsFrac = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DAQSummary")) {
    daqFrac = (dbe_->get(rootFolder_+"/EventInfo/DAQSummary"))->getFloatValue();
  }
  else daqFrac = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/reportSummary")) {
    reportFrac = (dbe_->get(rootFolder_+"/EventInfo/reportSummary"))->getFloatValue();
  }
  else reportFrac = -1;

  hcalFrac = 99.;
  hcalFrac = TMath::Min(hcalFrac,reportFrac);
  hcalFrac = TMath::Min(hcalFrac,daqFrac);
  hcalFrac = TMath::Min(hcalFrac,dcsFrac);
  if (debug_>0) {
    std::cout<<"dcsFrac= "<<dcsFrac<<std::endl;
    std::cout<<"daqFrac= "<<daqFrac<<std::endl;
    std::cout<<"reportFrac= "<<reportFrac<<std::endl;
    std::cout<<"CertificationSummary= "<<hcalFrac<<std::endl;
  }

  // reportSummary

  if (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HB")) {
    fracHCAL[0][0] = (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HB"))->getFloatValue();
  }
  else fracHCAL[0][0] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HE")) {
    fracHCAL[1][0] = (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HE"))->getFloatValue();
  }
  else fracHCAL[1][0] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO")) {
    fracHCAL[2][0] = (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO"))->getFloatValue();
  }
  else fracHCAL[2][0] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HF")) {
    fracHCAL[3][0] = (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HF"))->getFloatValue();
  }
  else fracHCAL[3][0] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO0")) {
    fracHCAL[4][0] = (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO0"))->getFloatValue();
  }
  else fracHCAL[4][0] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO12")) {
    fracHCAL[5][0] = (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO12"))->getFloatValue();
  }
  else fracHCAL[5][0] = -1;


  if (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HFlumi")) {
    fracHCAL[6][0] = (dbe_->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HFlumi"))->getFloatValue();
  }
  else fracHCAL[6][0] = -1;

  // DAQ

  if (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HB")) {
    fracHCAL[0][1] = (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HB"))->getFloatValue();
  }
  else fracHCAL[0][1] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HE")) {
    fracHCAL[1][1] = (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HE"))->getFloatValue();
  }
  else fracHCAL[1][1] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO")) {
    fracHCAL[2][1] = (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO"))->getFloatValue();
  }
  else fracHCAL[2][1] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HF")) {
    fracHCAL[3][1] = (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HF"))->getFloatValue();
  }
  else fracHCAL[3][1] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO0")) {
    fracHCAL[4][1] = (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO0"))->getFloatValue();
  }
  else fracHCAL[4][1] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO12")) {
    fracHCAL[5][1] = (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO12"))->getFloatValue();
  }
  else fracHCAL[5][1] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HFlumi")) {
    fracHCAL[6][1] = (dbe_->get(rootFolder_+"/EventInfo/DAQContents/Hcal_HFlumi"))->getFloatValue();
  }
  else fracHCAL[6][1] = -1;

  // DCS

  if (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HB")) {
    fracHCAL[0][2] = (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HB"))->getFloatValue();
  }
  else fracHCAL[0][2] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HE")) {
    fracHCAL[1][2] = (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HE"))->getFloatValue();
  }
  else fracHCAL[1][2] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO")) {
    fracHCAL[2][2] = (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO"))->getFloatValue();
  }
  else fracHCAL[2][2] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HF")) {
    fracHCAL[3][2] = (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HF"))->getFloatValue();
  }
  else fracHCAL[3][2] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO0")) {
    fracHCAL[4][2] = (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO0"))->getFloatValue();
  }
  else fracHCAL[4][2] = -1;

  if (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO12")) {
    fracHCAL[5][2] = (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO12"))->getFloatValue();
  }
  else fracHCAL[5][2] = -1;


  if (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HFlumi")) {
    fracHCAL[6][2] = (dbe_->get(rootFolder_+"/EventInfo/DCSContents/Hcal_HFlumi"))->getFloatValue();
  }
  else fracHCAL[6][2] = -1;

  for (int ii=0;ii<7;ii++) {
    certHcal[ii] = 99.0;
    for (int jj=0; jj<2;jj++) certHcal[ii] = TMath::Min(certHcal[ii],fracHCAL[ii][jj]);
    CertificationSummaryMap->setBinContent(ii+1,1,certHcal[ii]);
    if (debug_>0) std::cout<<"certFrac["<<ii<<"]= "<<certHcal[ii]<<std::endl;
  }

  CertificationSummary->Fill(hcalFrac);
  Hcal_HB->Fill(certHcal[0]);
  Hcal_HE->Fill(certHcal[1]);
  Hcal_HO->Fill(certHcal[2]);
  Hcal_HF->Fill(certHcal[3]);
  Hcal_HO0->Fill(certHcal[4]);
  Hcal_HO12->Fill(certHcal[5]);
  Hcal_HFlumi->Fill(certHcal[6]);

// ---------------------- end of certification
  if (debug_>0) std::cout << "<HcalDataCertification::MEfilled= " << std::endl;

}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDataCertification);
