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
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class HcalDAQInfo : public DQMEDHarvester {
   public:
      explicit HcalDAQInfo(const edm::ParameterSet&);
      ~HcalDAQInfo();

      virtual void dqmEndLuminosityBlock(DQMStore::IBooker &ib, DQMStore::IGetter &ig, edm::LuminosityBlock const &, edm::EventSetup const &);
      virtual void dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) { }

   // ----------member data ---------------------------

   edm::ParameterSet conf_;
   MonitorElement* HcalDaqFraction;
   MonitorElement* DAQSummaryMap;
   MonitorElement* HBDaqFraction;
   MonitorElement* HEDaqFraction;
   MonitorElement* HODaqFraction;
   MonitorElement* HFDaqFraction;
   MonitorElement* HO0DaqFraction;
   MonitorElement* HO12DaqFraction;
   MonitorElement* HFlumiDaqFraction;
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

HcalDAQInfo::HcalDAQInfo(const edm::ParameterSet& iConfig)
{
  // now do what ever initialization is needed
  debug_=iConfig.getUntrackedParameter<int>("debug",0);
  rootFolder_ = iConfig.getUntrackedParameter<std::string>("subSystemFolder","Hcal");

  HcalDaqFraction = NULL;

}

HcalDAQInfo::~HcalDAQInfo()
{ 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//


// ------------ method called right after a luminosity block ends ------------
void 
HcalDAQInfo::dqmEndLuminosityBlock(DQMStore::IBooker &ib, DQMStore::IGetter &ig, const edm::LuminosityBlock& run, const edm::EventSetup& iSetup)
{
  if (debug_>0) {
    std::cout <<"<HcalDAQInfo::endLuminosityBlock> "<<std::endl;
  }

  // check that MonitorElements exist
  // if not, book the histograms
  if ( !HcalDaqFraction ) {
    ib.setCurrentFolder(rootFolder_);
    std::string currDir = ib.pwd();
    if (debug_>0) std::cout << "--- Current Directory " << currDir << std::endl;
  
    ib.setCurrentFolder(rootFolder_+"/EventInfo/");
  
    HcalDaqFraction = ib.bookFloat("DAQSummary");
  
    DAQSummaryMap = ib.book2D("DAQSummaryMap","HcalDAQSummaryMap",7,0.,7.,1,0.,1.);
    DAQSummaryMap->setAxisRange(-1,1,3);
    DAQSummaryMap->setBinLabel(1,"HB");
    DAQSummaryMap->setBinLabel(2,"HE");
    DAQSummaryMap->setBinLabel(3,"HO");
    DAQSummaryMap->setBinLabel(4,"HF");
    DAQSummaryMap->setBinLabel(5,"H00");
    DAQSummaryMap->setBinLabel(6,"H012");
    DAQSummaryMap->setBinLabel(7,"HFlumi");
    DAQSummaryMap->setBinLabel(1,"Status",2);
  
    ib.setCurrentFolder(rootFolder_+"/EventInfo/DAQContents/");
    HBDaqFraction  = ib.bookFloat("Hcal_HB");
    HEDaqFraction  = ib.bookFloat("Hcal_HE");
    HODaqFraction  = ib.bookFloat("Hcal_HO");
    HFDaqFraction  = ib.bookFloat("Hcal_HF");
    HO0DaqFraction = ib.bookFloat("Hcal_HO0");
    HO12DaqFraction   = ib.bookFloat("Hcal_HO12");
    HFlumiDaqFraction = ib.bookFloat("Hcal_HFlumi");
  }

  HcalDaqFraction->Fill(-1);

  for (int ii=0; ii<7; ii++) DAQSummaryMap->setBinContent(ii+1,1,-1);

  HBDaqFraction->Fill(-1);
  HEDaqFraction->Fill(-1);
  HODaqFraction->Fill(-1);
  HFDaqFraction->Fill(-1);
  HO0DaqFraction->Fill(-1);
  HO12DaqFraction->Fill(-1);
  HFlumiDaqFraction->Fill(-1);

  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));

  if( iSetup.find( recordKey ) ) {

    edm::ESHandle<RunInfo> sumFED;
    iSetup.get<RunInfoRcd>().get(sumFED);    
   
    std::vector<int> FedsInIds= sumFED->m_fed_in;   

    float HcalFedCount   = 0.;
    float HBFedCount     = 0.;
    float HEFedCount     = 0.;
    float HFFedCount     = 0.;
    float HOFedCount     = 0.;
    float HO0FedCount    = 0.;
    float HO12FedCount   = 0.;
    float HFlumiFedCount = 0.;

    // By FED taking into account Nchannels per FED

    for( unsigned int fedItr=0; fedItr<FedsInIds.size(); ++fedItr ) {

      int fedID=FedsInIds[fedItr];

      if (fedID >= 700 && fedID <= 731) {
        HcalFedCount++;
	if (fedID >= 700 && fedID <= 717)  {
	  HBFedCount++;
	  HEFedCount++;
	}
	else if (fedID >= 718 && fedID <= 723)  {
	  HFFedCount++;
	  HFlumiFedCount++;
	}
	else if (fedID >= 724 && fedID <= 731)  {
	  if (fedID%2 == 0) {
	    HOFedCount += 276;
	    HO0FedCount += 84;
	    HO12FedCount += 192;
	  }
	  else {
	    HOFedCount += 264;
	    HO0FedCount += 60;
	    HO12FedCount += 204;
	  }
	}
      }

      //else if ( fedID == 735 ) std::cout<<fedID<<" -- LumiScaler"<<std::endl;   
    }

    HcalFedCount = (HBFedCount*144+HEFedCount*144+HFFedCount*288+HOFedCount)/9072;
    HBFedCount /= 18;
    HEFedCount /= 18;
    HFFedCount /= 6;
    HFlumiFedCount /= 6;
    HOFedCount /= 2160;
    HO0FedCount /= 576;
    HO12FedCount /= 1584;

    DAQSummaryMap->setBinContent(1,1,HBFedCount);
    DAQSummaryMap->setBinContent(2,1,HEFedCount);
    DAQSummaryMap->setBinContent(3,1,HOFedCount);
    DAQSummaryMap->setBinContent(4,1,HFFedCount);
    DAQSummaryMap->setBinContent(5,1,HO0FedCount);
    DAQSummaryMap->setBinContent(6,1,HO12FedCount);
    DAQSummaryMap->setBinContent(7,1,HFlumiFedCount);

    HcalDaqFraction->Fill(HcalFedCount);
    HBDaqFraction->Fill(HBFedCount);
    HEDaqFraction->Fill(HEFedCount);
    HFDaqFraction->Fill(HFFedCount);
    HODaqFraction->Fill(HOFedCount);
    HO0DaqFraction->Fill(HO0FedCount);
    HO12DaqFraction->Fill(HO12FedCount);
    HFlumiDaqFraction->Fill(HFlumiFedCount);

    if (debug_>0) {
      std::cout<<" HcalFedCount= "<<HcalFedCount<<std::endl;
      std::cout<<" HBFedCount= "<<HBFedCount<<std::endl;
      std::cout<<" HEFedCount= "<<HEFedCount<<std::endl;
      std::cout<<" HFFedCount= "<<HFFedCount<<std::endl;
      std::cout<<" HOFedCount= "<<HOFedCount<<std::endl;
      std::cout<<" HO0FedCount= "<<HO0FedCount<<std::endl;
      std::cout<<" HO12FedCount= "<<HO12FedCount<<std::endl;
      std::cout<<" HFlumiFedCount= "<<HFlumiFedCount<<std::endl;
    }
  }
  else edm::LogInfo(rootFolder_+"/EventInfo/")<<"No RunInfoRcd"<<std::endl;

// ---------------------- end of DAQ Info
  if (debug_>0) std::cout << "HcalDAQInfo::MEfilled " << std::endl;

}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDAQInfo);
