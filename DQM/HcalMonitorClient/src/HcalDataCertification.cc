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
// $Id: HcalDataCertification.cc,v 1.6 2009/08/24 11:23:20 temple Exp $
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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class HcalDataCertification : public edm::EDAnalyzer {
   public:
      explicit HcalDataCertification(const edm::ParameterSet&);
      ~HcalDataCertification();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) ;
      virtual void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) ;

   // ----------member data ---------------------------

   edm::ParameterSet conf_;
   DQMStore * dbe;
   edm::Service<TFileService> fs_;
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

HcalDataCertification::HcalDataCertification(const edm::ParameterSet& iConfig):conf_(iConfig)
{
  // now do what ever initialization is needed
  debug_ = iConfig.getUntrackedParameter<int>("debug",0);
  rootFolder_ = iConfig.getUntrackedParameter<std::string>("subSystemFolder","Hcal");
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
HcalDataCertification::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalDataCertification::beginJob(const edm::EventSetup&)
{
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  if (debug_>0) std::cout<<"<HcalDataCertification> beginJob"<< std::endl;
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

  float dcsFrac,daqFrac,fracHB,fracHE,fracHF,fracHO;
  float fracHO0, fracHO12, fracHFlumi;

  dbe->setCurrentFolder(rootFolder_);
  std::string currDir = dbe->pwd();
  if (debug_>0) std::cout << "<HcalDataCertification::endLuminosityBlock> --- Current Directory " << currDir << std::endl;
  std::vector<MonitorElement*> mes = dbe->getAllContents("");
  if (debug_>0) std::cout << "<HcalDataCertification::endLuminosityBlock> found " << mes.size() << " monitoring elements:" << std::endl;

  dbe->setCurrentFolder(rootFolder_+"/EventInfo/CertificationContents/");
  MonitorElement* Hcal_HB = dbe->bookFloat("Hcal_HB");
  MonitorElement* Hcal_HE = dbe->bookFloat("Hcal_HE");
  MonitorElement* Hcal_HF = dbe->bookFloat("Hcal_HF");
  MonitorElement* Hcal_HO = dbe->bookFloat("Hcal_HO");
  MonitorElement* Hcal_HFlumi = dbe->bookFloat("Hcal_HFlumi");
  MonitorElement* Hcal_HO0    = dbe->bookFloat("HcalHO0");
  MonitorElement* Hcal_HO12   = dbe->bookFloat("HcalHO12");
  Hcal_HB->Fill(-1);
  Hcal_HE->Fill(-1);
  Hcal_HF->Fill(-1);
  Hcal_HO->Fill(-1);
  Hcal_HO0->Fill(-1);
  Hcal_HO12->Fill(-1);
  Hcal_HFlumi->Fill(-1);

  int nevt = (dbe->get(rootFolder_+"/EventInfo/processedEvents"))->getIntValue();
  if (debug_>0) std::cout << "<HcalDataCertification::nevt= " << nevt << std::endl;
  if (debug_>0 && nevt<1) {
    edm::LogInfo(rootFolder_+"DataCertification")<<"Nevents processed ="<<nevt<<" => exit"<<std::endl;
    return;
  }

  if (dbe->get(rootFolder_+"/EventInfo/DCSContents/HcalDcsFraction")) {
    //dcsFrac = (dbe->get(rootFolder_+"/EventInfo/DCSContents/HcalDcsFraction"))->getFloatValue();
    dcsFrac = (dbe->get(rootFolder_+"/EventInfo/DCSSummary"))->getFloatValue();
  }
  else {
    dcsFrac = -1;
    edm::LogInfo(rootFolder_+"DataCertification")<<"No DCS info"<<std::endl;
  }
  if (dbe->get(rootFolder_+"/EventInfo/DAQContents/HcalDaqFraction")) {
    daqFrac = (dbe->get(rootFolder_+"/EventInfo/DAQContents/HcalDaqFraction"))->getFloatValue();
    daqFrac = (dbe->get(rootFolder_+"/EventInfo/DAQSummary"))->getFloatValue();
  }
  else  {
    daqFrac = -1;
    edm::LogInfo(rootFolder_+"DataCertification")<<"No DAQ info"<<std::endl;
  }

  // Get DQM Report Summary information (.../EventInfo/reportSummaryConents/Hcal_...)
  if (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HB")) {
    fracHB = (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HB"))->getFloatValue();
  }
  else   {
    fracHB =-1;
    edm::LogInfo(rootFolder_+"DataCertification")<<"No Hcal_HB ME"<<std::endl;
  }
  if (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HE")) {
    fracHE = (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HE"))->getFloatValue();
  }
  else   {
    fracHE = -1;
    edm::LogInfo(rootFolder_+"DataCertification")<<"No Hcal_HE ME"<<std::endl;
  }
  if (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HF")) {
    fracHF = (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HF"))->getFloatValue();
  }
  else   {
    fracHF = -1;
    edm::LogInfo(rootFolder_+"DataCertification")<<"No Hcal_HF ME"<<std::endl;
  }
  if (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO")) {
    fracHO = (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO"))->getFloatValue();
  }
  else   {
    fracHO = -1;
    edm::LogInfo(rootFolder_+"DataCertification")<<"No Hcal_HO ME"<<std::endl;
 }
  if (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO0")) {
    fracHO0 = (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO0"))->getFloatValue();
  }
  else   {
    fracHO0 = -1;
    edm::LogInfo(rootFolder_+"DataCertification")<<"No Hcal_HO0 ME"<<std::endl;
  }
  if (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO12")) {
    fracHO12 = (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HO12"))->getFloatValue();
  }
  else   {
    fracHO12 = -1;
    edm::LogInfo(rootFolder_+"DataCertification")<<"No Hcal_HO12 ME"<<std::endl;
  }
  if (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HFlumi")) {
    fracHFlumi = (dbe->get(rootFolder_+"/EventInfo/reportSummaryContents/Hcal_HFlumi"))->getFloatValue();
  }
  else   {
    fracHFlumi = -1;
    edm::LogInfo(rootFolder_+"DataCertification")<<"No Hcal_HFlumi ME"<<std::endl;
  }


  Hcal_HB->Fill(fracHB);
  Hcal_HE->Fill(fracHE);
  Hcal_HF->Fill(fracHF);
  Hcal_HO->Fill(fracHO);
  Hcal_HO0->Fill(fracHO0);
  Hcal_HO12->Fill(fracHO12);
  Hcal_HFlumi->Fill(fracHFlumi);

// ---------------------- end of certification
  if (debug_>0) std::cout << "<HcalDataCertification::MEfilled= " << std::endl;

}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDataCertification);
