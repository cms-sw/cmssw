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
#include "DQMServices/Core/interface/DQMEDHarvester.h"

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

class HcalDataCertification : public DQMEDHarvester {
   public:
      explicit HcalDataCertification(const edm::ParameterSet&);
      ~HcalDataCertification();
      
    virtual void dqmEndLuminosityBlock(DQMStore::IBooker &ib, DQMStore::IGetter & ig, const edm::LuminosityBlock&, const edm::EventSetup&) override ;
    virtual void dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig ) override;

   private:
  void CertifyHcal(DQMStore::IBooker &, DQMStore::IGetter &);

  // helper method to book histograms
  void bookHistograms(DQMStore::IBooker &);

   // ----------member data ---------------------------

   edm::ParameterSet conf_;
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

  CertificationSummary = NULL;
}

HcalDataCertification::~HcalDataCertification()
{ 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void 
HcalDataCertification::bookHistograms(DQMStore::IBooker &ib)
{
  if (debug_>0) std::cout<<"<HcalDataCertification> bookHistograms"<< std::endl;

  ib.setCurrentFolder(rootFolder_);
  std::string currDir = ib.pwd();

  ib.setCurrentFolder(rootFolder_+"/EventInfo/");

  CertificationSummary = ib.bookFloat("CertificationSummary");

  CertificationSummaryMap = ib.book2D("CertificationSummaryMap","HcalCertificationSummaryMap",7,0.,7.,1,0.,1.);
  CertificationSummaryMap->setAxisRange(-1,1,3);
  CertificationSummaryMap->setBinLabel(1,"HB");
  CertificationSummaryMap->setBinLabel(2,"HE");
  CertificationSummaryMap->setBinLabel(3,"HO");
  CertificationSummaryMap->setBinLabel(4,"HF");
  CertificationSummaryMap->setBinLabel(5,"H00");
  CertificationSummaryMap->setBinLabel(6,"H012");
  CertificationSummaryMap->setBinLabel(7,"HFlumi");
  CertificationSummaryMap->setBinLabel(1,"Status",2);

  ib.setCurrentFolder(rootFolder_+"/EventInfo/CertificationContents/");
  Hcal_HB = ib.bookFloat("Hcal_HB");
  Hcal_HE = ib.bookFloat("Hcal_HE");
  Hcal_HF = ib.bookFloat("Hcal_HF");
  Hcal_HO = ib.bookFloat("Hcal_HO");
  Hcal_HFlumi = ib.bookFloat("Hcal_HFlumi");
  Hcal_HO0    = ib.bookFloat("Hcal_HO0");
  Hcal_HO12   = ib.bookFloat("Hcal_HO12");

}



// ------------ method called right after a run ends ------------
void 
HcalDataCertification::dqmEndLuminosityBlock(DQMStore::IBooker &ib, DQMStore::IGetter &ig, const edm::LuminosityBlock& run, const edm::EventSetup& c)
{
  // check if MonitorElements exits
  // book them if not
  if ( !CertificationSummary ) {
    bookHistograms(ib);
  }
  CertifyHcal(ib,ig);
}

// this used to be endRun, after migration I make it dqmEndJob
void HcalDataCertification::dqmEndJob(DQMStore::IBooker & ib, DQMStore::IGetter &ig )
{
  // check if MonitorElements exits
  // book them if not
  if ( !CertificationSummary ) {
    bookHistograms(ib);
  }
  CertifyHcal(ib,ig);
}

void HcalDataCertification::CertifyHcal(DQMStore::IBooker &ib, DQMStore::IGetter &ig)
{

  float hcalFrac,reportFrac,dcsFrac,daqFrac;
  float fracHCAL[7][3];
  float certHcal[7];

  if (debug_>0) {
    ig.setCurrentFolder(rootFolder_);
    std::string currDir = ib.pwd();
    std::cout << "<HcalDataCertification::endLuminosityBlock> --- Current Directory " << currDir << std::endl;
  }

  if (ig.get(rootFolder_+"/EventInfo/DCSSummary")) {
    dcsFrac = (ig.get(rootFolder_+"/EventInfo/DCSSummary"))->getFloatValue();
  }
  else dcsFrac = -1;

  if (ig.get(rootFolder_+"/EventInfo/DAQSummary")) {
    daqFrac = (ig.get(rootFolder_+"/EventInfo/DAQSummary"))->getFloatValue();
  }
  else daqFrac = -1;

  if (ig.get(rootFolder_+"/EventInfo/reportSummary")) {
    reportFrac = (ig.get(rootFolder_+"/EventInfo/reportSummary"))->getFloatValue();
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

  if (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HB")) {
    fracHCAL[0][0] = (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HB"))->getFloatValue();
  }
  else fracHCAL[0][0] = -1;

  if (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HE")) {
    fracHCAL[1][0] = (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HE"))->getFloatValue();
  }
  else fracHCAL[1][0] = -1;

  if (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO")) {
    fracHCAL[2][0] = (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO"))->getFloatValue();
  }
  else fracHCAL[2][0] = -1;

  if (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HF")) {
    fracHCAL[3][0] = (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HF"))->getFloatValue();
  }
  else fracHCAL[3][0] = -1;

  if (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO0")) {
    fracHCAL[4][0] = (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO0"))->getFloatValue();
  }
  else fracHCAL[4][0] = -1;

  if (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO12")) {
    fracHCAL[5][0] = (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO12"))->getFloatValue();
  }
  else fracHCAL[5][0] = -1;


  if (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HFlumi")) {
    fracHCAL[6][0] = (ig.get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HFlumi"))->getFloatValue();
  }
  else fracHCAL[6][0] = -1;

  // DAQ

  if (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HB")) {
    fracHCAL[0][1] = (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HB"))->getFloatValue();
  }
  else fracHCAL[0][1] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HE")) {
    fracHCAL[1][1] = (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HE"))->getFloatValue();
  }
  else fracHCAL[1][1] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO")) {
    fracHCAL[2][1] = (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO"))->getFloatValue();
  }
  else fracHCAL[2][1] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HF")) {
    fracHCAL[3][1] = (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HF"))->getFloatValue();
  }
  else fracHCAL[3][1] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO0")) {
    fracHCAL[4][1] = (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO0"))->getFloatValue();
  }
  else fracHCAL[4][1] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO12")) {
    fracHCAL[5][1] = (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HO12"))->getFloatValue();
  }
  else fracHCAL[5][1] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HFlumi")) {
    fracHCAL[6][1] = (ig.get(rootFolder_+"/EventInfo/DAQContents/Hcal_HFlumi"))->getFloatValue();
  }
  else fracHCAL[6][1] = -1;

  // DCS

  if (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HB")) {
    fracHCAL[0][2] = (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HB"))->getFloatValue();
  }
  else fracHCAL[0][2] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HE")) {
    fracHCAL[1][2] = (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HE"))->getFloatValue();
  }
  else fracHCAL[1][2] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO")) {
    fracHCAL[2][2] = (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO"))->getFloatValue();
  }
  else fracHCAL[2][2] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HF")) {
    fracHCAL[3][2] = (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HF"))->getFloatValue();
  }
  else fracHCAL[3][2] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO0")) {
    fracHCAL[4][2] = (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO0"))->getFloatValue();
  }
  else fracHCAL[4][2] = -1;

  if (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO12")) {
    fracHCAL[5][2] = (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HO12"))->getFloatValue();
  }
  else fracHCAL[5][2] = -1;


  if (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HFlumi")) {
    fracHCAL[6][2] = (ig.get(rootFolder_+"/EventInfo/DCSContents/Hcal_HFlumi"))->getFloatValue();
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
