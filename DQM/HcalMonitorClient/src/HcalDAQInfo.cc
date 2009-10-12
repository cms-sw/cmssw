// -*- C++ -*-
//
// Package:    DQMO/HcalMonitorClient/HcalDAQInfo
// Class:      HcalDAQInfo
// 
/**\class HcalDAQInfo HcalDAQInfo.cc DQM/HcalMonitorClient/src/HcalDAQInfo.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Igor Vodopiyanov"
//         Created:  Feb-21 2009
//
//


// system include files
#include <memory>
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <exception>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/HcalMonitorTasks/interface/HcalDataFormatMonitor.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class HcalDAQInfo : public edm::EDAnalyzer {
   public:
      explicit HcalDAQInfo(const edm::ParameterSet&);
      ~HcalDAQInfo();

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

HcalDAQInfo::HcalDAQInfo(const edm::ParameterSet& iConfig):conf_(iConfig)
{
  // now do what ever initialization is needed
  debug_=iConfig.getUntrackedParameter<int>("debug",0);
}


HcalDAQInfo::~HcalDAQInfo()
{ 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HcalDAQInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalDAQInfo::beginJob(const edm::EventSetup&)
{
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  if (debug_>0) std::cout<<"<HcalDAQInfo::beginJob>"<< std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalDAQInfo::endJob() 
{
  if (debug_>0) std::cout << "<HcalDAQInfo::endJob> " << std::endl;
}

// ------------ method called just before starting a new run  ------------
void 
HcalDAQInfo::beginLuminosityBlock(const edm::LuminosityBlock& run, const edm::EventSetup& c)
{
  if (debug_>0) std::cout<<"<HcalDAQInfo::beginLuminosityBlock>"<<std::endl;
}

// ------------ method called right after a run ends ------------
void 
HcalDAQInfo::endLuminosityBlock(const edm::LuminosityBlock& run, const edm::EventSetup& c)
{
  if (debug_>0) std::cout <<"<HcalDAQInfo::endLuminosityBlock> "<<endl;
  dbe->setCurrentFolder("Hcal");
  std::string currDir = dbe->pwd();
  if (debug_>0) std::cout << "--- Current Directory " << currDir << std::endl;
  std::vector<MonitorElement*> mes = dbe->getAllContents("");
  if (debug_>0) std::cout << "found " << mes.size() << " monitoring elements:" << std::endl;

  //dbe->setCurrentFolder("Hcal/EventInfo/DAQContents/");
  //MonitorElement* HcalDaqFraction = dbe->bookFloat("Hcal");

  dbe->setCurrentFolder("Hcal/EventInfo/");
  MonitorElement* HcalDaqFraction = dbe->bookFloat("DAQSummary");
  HcalDaqFraction->Fill(-1);

  dbe->setCurrentFolder("Hcal/EventInfo/DAQSummaryContents/");
  MonitorElement* HBDaqFraction = dbe->bookFloat("Hcal_HB");
  MonitorElement* HEDaqFraction = dbe->bookFloat("Hcal_HE");
  MonitorElement* HODaqFraction = dbe->bookFloat("Hcal_HO");
  MonitorElement* HFDaqFraction = dbe->bookFloat("Hcal_HF");
  MonitorElement* ZDCDaqFraction = dbe->bookFloat("Hcal_ZDC");
  
  HBDaqFraction->Fill(-1);
  HEDaqFraction->Fill(-1);
  HODaqFraction->Fill(-1);
  HFDaqFraction->Fill(-1);
  ZDCDaqFraction->Fill(-1);

  int nevt = (dbe->get("Hcal/EventInfo/processedEvents"))->getIntValue();
  if (debug_>0) std::cout << "HcalDAQInfo::nevt= " << nevt << std::endl;
  if (nevt<1) {
    edm::LogInfo("HcalDAQInfo")<<"Nevents processed ="<<nevt<<" => exit"<<std::endl;
    return;
  }

  TH1F *hFEDEntries;
  float nFEDEntries;

  if (dbe->get("Hcal/DataFormatMonitor/HcalFEDChecking/FEDEntries")) {
    hFEDEntries = (dbe->get("Hcal/DataFormatMonitor/HcalFEDChecking/FEDEntries"))->getTH1F();
    nFEDEntries = hFEDEntries->Integral(); 
    HcalDaqFraction->Fill(nFEDEntries/(32.0*nevt));
    //Subdetector FEDs:  HB/HE:  700-717, HF:  718-723, HO:  724-731
    // (Integral(...) uses BinContent; starts with bin 1)
    HBDaqFraction->Fill(hFEDEntries->Integral(1,18)/(18.*nevt));
    HEDaqFraction->Fill(hFEDEntries->Integral(1,18)/(18.*nevt));
    HFDaqFraction->Fill(hFEDEntries->Integral(19,24)/(6.*nevt));
    HODaqFraction->Fill(hFEDEntries->Integral(25,32)/(8.*nevt));
  }
  else {
    edm::LogInfo("HcalDAQInfo")<<"No DAQ info"<<std::endl;
  }



// ---------------------- end of DAQ Info
  if (debug_>0) std::cout << "HcalDAQInfo::MEfilled " << std::endl;

}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDAQInfo);
