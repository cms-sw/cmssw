#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/src/fed_header.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"

#include "DQM/EcalPreshowerMonitorModule/interface/ESRawDataTask.h"

using namespace cms;
using namespace edm;
using namespace std;

ESRawDataTask::ESRawDataTask(const ParameterSet& ps) {

   prefixME_      = ps.getUntrackedParameter<string>("prefixME", "");

   FEDRawDataCollection_ = consumes<FEDRawDataCollection>(ps.getParameter<InputTag>("FEDRawDataCollection"));
   dccCollections_       = consumes<ESRawDataCollection>(ps.getParameter<InputTag>("ESDCCCollections"));

   ievt_ = 0;
}

void ESRawDataTask::bookHistograms(DQMStore::IBooker& iBooker, Run const&, EventSetup const&)
{
   char histo[200];

   iBooker.setCurrentFolder(prefixME_ + "/ESRawDataTask");

   //sprintf(histo, "ES run number errors");
   //meRunNumberErrors_ = iBooker.book1D(histo, histo, 56, 519.5, 575.5); 
   //meRunNumberErrors_->setAxisTitle("ES FED", 1);
   //meRunNumberErrors_->setAxisTitle("Num of Events", 2);

   sprintf(histo, "ES L1A DCC errors");
   meL1ADCCErrors_ = iBooker.book1D(histo, histo, 56, 519.5, 575.5); 
   meL1ADCCErrors_->setAxisTitle("ES FED", 1);
   meL1ADCCErrors_->setAxisTitle("Num of Events", 2);

   sprintf(histo, "ES BX DCC errors");
   meBXDCCErrors_ = iBooker.book1D(histo, histo, 56, 519.5, 575.5); 
   meBXDCCErrors_->setAxisTitle("ES FED", 1);
   meBXDCCErrors_->setAxisTitle("Num of Events", 2);

   sprintf(histo, "ES Orbit Number DCC errors");
   meOrbitNumberDCCErrors_ = iBooker.book1D(histo, histo, 56, 519.5, 575.5); 
   meOrbitNumberDCCErrors_->setAxisTitle("ES FED", 1);
   meOrbitNumberDCCErrors_->setAxisTitle("Num of Events", 2);
      
   sprintf(histo, "Difference between ES and GT L1A");
   meL1ADiff_ = iBooker.book1D(histo, histo, 201, -100.5, 100.5);
   meL1ADiff_->setAxisTitle("ES - GT L1A", 1);
   meL1ADiff_->setAxisTitle("Num of Events", 2);

   sprintf(histo, "Difference between ES and GT BX");
   meBXDiff_ = iBooker.book1D(histo, histo, 201, -100.5, 100.5);
   meBXDiff_->setAxisTitle("ES - GT BX", 1);
   meBXDiff_->setAxisTitle("Num of Events", 2);

   sprintf(histo, "Difference between ES and GT Orbit Number");
   meOrbitNumberDiff_ = iBooker.book1D(histo, histo, 201, -100.5, 100.5);
   meOrbitNumberDiff_->setAxisTitle("ES - GT orbit number", 1);
   meOrbitNumberDiff_->setAxisTitle("Num of Events", 2);
}

void ESRawDataTask::endJob(void){

   LogInfo("ESRawDataTask") << "analyzed " << ievt_ << " events";

}

void ESRawDataTask::analyze(const Event& e, const EventSetup& c){

   ievt_++;
   runNum_ = e.id().run();

   int gt_L1A = 0, gt_OrbitNumber = 0, gt_BX = 0;
   int esDCC_L1A_MostFreqCounts = 0;
   int esDCC_BX_MostFreqCounts = 0;
   int esDCC_OrbitNumber_MostFreqCounts = 0;

   Handle<ESRawDataCollection> dccs;
   Handle<FEDRawDataCollection> allFedRawData;

   int gtFedDataSize = 0;
   
   if ( e.getByToken(FEDRawDataCollection_, allFedRawData) ) {
     
     // GT FED data
     const FEDRawData& gtFedData = allFedRawData->FEDData(812);
     
     gtFedDataSize = gtFedData.size()/sizeof(uint64_t);
     
     if ( gtFedDataSize > 0 ) {
       
       FEDHeader header(gtFedData.data());
       
       gt_L1A         = header.lvl1ID();
       gt_OrbitNumber = e.orbitNumber();
       gt_BX          = e.bunchCrossing();
     } else {

       map<int, int> esDCC_L1A_FreqMap;
       map<int, int> esDCC_BX_FreqMap;
       map<int, int> esDCC_OrbitNumber_FreqMap;

       if ( e.getByToken(dccCollections_, dccs) ) {
	 for (ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr) {
	   ESDCCHeaderBlock esdcc = (*dccItr);
	   
	   esDCC_L1A_FreqMap[esdcc.getLV1()]++;
	   esDCC_BX_FreqMap[esdcc.getBX()]++;
	   esDCC_OrbitNumber_FreqMap[esdcc.getOrbitNumber()]++;
	   
	   if (esDCC_L1A_FreqMap[esdcc.getLV1()] > esDCC_L1A_MostFreqCounts) {
	     esDCC_L1A_MostFreqCounts = esDCC_L1A_FreqMap[esdcc.getLV1()];
	     gt_L1A = esdcc.getLV1();
	   } 
	   
	   if (esDCC_BX_FreqMap[esdcc.getBX()] > esDCC_BX_MostFreqCounts) {
	     esDCC_BX_MostFreqCounts = esDCC_BX_FreqMap[esdcc.getBX()];
	     gt_BX = esdcc.getBX();
	   } 
	   
	   if (esDCC_OrbitNumber_FreqMap[esdcc.getOrbitNumber()] > esDCC_OrbitNumber_MostFreqCounts) {
	     esDCC_OrbitNumber_MostFreqCounts = esDCC_OrbitNumber_FreqMap[esdcc.getOrbitNumber()];
	     gt_OrbitNumber = esdcc.getOrbitNumber();
	   } 
	   
	 }
       } else {
	 LogWarning("ESRawDataTask") << "dccCollections not available";
       }

     }
   } else {
     LogWarning("ESRawDataTask") << "FEDRawDataCollection not available";
   }

   // DCC 
   vector<int> fiberStatus;
   if ( e.getByToken(dccCollections_, dccs) ) {
     
     for (ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr) {
       ESDCCHeaderBlock dcc = (*dccItr);
       
       //if (dcc.getRunNumber() != runNum_) {
       //meRunNumberErrors_->Fill(dcc.fedId());
       //cout<<"Run # err : "<<dcc.getRunNumber()<<" "<<runNum_<<endl;
       //}
       
       if (dcc.getLV1() != gt_L1A) {
	 meL1ADCCErrors_->Fill(dcc.fedId());
	 //cout<<"L1A err : "<<dcc.getLV1()<<" "<<gt_L1A<<endl;
	 Float_t l1a_diff = dcc.getLV1() - gt_L1A;
	 if (l1a_diff > 100) l1a_diff = 100;
	 else if (l1a_diff < -100) l1a_diff = -100;
	 meL1ADiff_->Fill(l1a_diff);
       }
       
       if (dcc.getBX() != gt_BX) {
	 meBXDCCErrors_->Fill(dcc.fedId());
	 //cout<<"BX err : "<<dcc.getBX()<<" "<<gt_BX<<endl;
	 Float_t bx_diff = dcc.getBX() - gt_BX;
	 if (bx_diff > 100) bx_diff = 100;
	 else if (bx_diff < -100) bx_diff = -100;
	 meBXDiff_->Fill(bx_diff);
       }
       if (dcc.getOrbitNumber() != gt_OrbitNumber) {
	 meOrbitNumberDCCErrors_->Fill(dcc.fedId());
	 //cout<<"Orbit err : "<<dcc.getOrbitNumber()<<" "<<gt_OrbitNumber<<endl;
	 Float_t orbitnumber_diff = dcc.getOrbitNumber() - gt_OrbitNumber;
	 if (orbitnumber_diff > 100) orbitnumber_diff = 100;
	 else if (orbitnumber_diff < -100) orbitnumber_diff = -100;
	 meOrbitNumberDiff_->Fill(orbitnumber_diff);
       }
     }
   }

}

DEFINE_FWK_MODULE(ESRawDataTask);
