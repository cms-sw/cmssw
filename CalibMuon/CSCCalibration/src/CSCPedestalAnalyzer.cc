/** 
 * Analyzer for reading CSC pedestals.
 * author O.Boeriu 18/03/06 
 * ripped from Jeremy's and Rick's analyzers
 *   
 */
#include <iostream>
#include <fstream>
#include <vector>
#include "string"

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "IORawData/CSCCommissioning/src/FileReaderDDU.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "CalibMuon/CSCCalibration/interface/CSCPedestalAnalyzer.h"

CSCPedestalAnalyzer::CSCPedestalAnalyzer(edm::ParameterSet const& conf) {
  
  eventNumber = 0;
  evt = 0;
  pedMean=0.0,time=0.0,max =-9999999.,max1=-9999999.;
  pedSum = 0, strip =-999,misMatch=0;
  i_chamber=0,i_layer=0,reportedChambers =0;
  aPeak=0.0,sumFive=0.0;
  length = 1, NChambers=0;
  
  for(int i=0;i<CHAMBERS;i++){
    for(int j=0; j<LAYERS; j++){
      for(int k=0; k<STRIPS; k++){
	arrayOfPed[i][j][k]       = 0.;
	arrayOfPedSquare[i][j][k] = 0.;
	arrayPed[i][j][k]         = 0.;
	arrayPeak[i][j][k]        = 0.;
	arrayOfPeak[i][j][k]      = 0.; 
	arrayOfPeakSquare[i][j][k]= 0.;
	arraySumFive[i][j][k]     = 0.;
      }
    }
  }

  for (int i=0;i<480;i++){
    newPed[i]=0;
    newRMS[i]=0;
    newPeakRMS[i]=0.;
    newPeak[i]=0.;
    newSumFive[i]=0.;
  }
}

void CSCPedestalAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
   edm::Handle<CSCStripDigiCollection> strips;
   
  // Pass the handle to the method "getByType", which is used to retrieve
  // one and only one instance of the type in question out of event "e". If
  // zero or more than one instance exists in the event an exception is thrown.
  //

   e.getByLabel("cscunpacker","MuonCSCStripDigi",strips);

   edm::Handle<FEDRawDataCollection> rawdata;
   e.getByLabel("DaqSource" , rawdata);

   for (int id=FEDNumbering::getCSCFEDIds().first;
	id<=FEDNumbering::getCSCFEDIds().second; ++id){ //for each of our DCCs
     
     /// Take a reference to this FED's data
     const FEDRawData& fedData = rawdata->FEDData(id);
     
     
     if (fedData.size()){ ///unpack data 
       
       ///get a pointer to data and pass it to constructor for unpacking
       CSCDCCEventData dccData((short unsigned int *) fedData.data()); 
       
       const std::vector<CSCDDUEventData> & dduData = dccData.dduData(); 
       
       for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) {  ///loop over DDUs
	 
	 ///get a reference to chamber data
	 const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	 
	 reportedChambers += dduData[iDDU].header().ncsc();
	 NChambers = cscData.size();
	 int repChambers = dduData[iDDU].header().ncsc();
	 std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	 if (NChambers!=repChambers) { std::cout<< "misMatched size!!!" << std::endl; misMatch++;}
	 
	 evt++;
	 
	 for (i_chamber=0; i_chamber<NChambers; i_chamber++) {//loop over all DMBs  
	   
	   for(i_layer = 1; i_layer <= 6; ++i_layer) {//loop over all layers in chambers
	     
	     std::vector<CSCStripDigi> digis = cscData[i_chamber].stripDigis(i_layer) ;
	     const CSCDMBHeader &thisDMBheader = cscData[i_chamber].dmbHeader();
	     
	     if (thisDMBheader.cfebAvailable()){//check that CFEB data exists
	       
	       dmbID[i_chamber]   = cscData[i_chamber].dmbHeader().dmbID(); //get DMB ID
	       crateID[i_chamber] = cscData[i_chamber].dmbHeader().crateID(); //get crate ID
	       if(crateID[i_chamber] == 255) continue; //255 is reserved for old crate, present only 0 and 1
	       
	       for (unsigned int i=0; i<digis.size(); i++){//loop over digis
		 strip = digis[i].getStrip();
		 adc   = digis[i].getADCCounts();
		 
		 pedSum  = adc[0]+adc[1];
		 pedMean = pedSum/2.0;
		 
		 for(unsigned int k=0;k<adc.size();k++){//loop over timeBins
		   time = (50. * k)-((evt%20)* 6.25)+116.5;	      
		   aPeak = adc[3];
		   if (max < aPeak) {
		     max = aPeak;
		   }
		   sumFive = adc[2]+adc[3]+adc[4];
		   
		   if (max1<sumFive){
		     max1=sumFive;
		   }
	    
		 }//adc.size
		 
		 arrayPed[i_chamber][i_layer-1][strip-1] = pedMean;
		 arrayOfPed[i_chamber][i_layer - 1][strip - 1] += pedMean;
		 arrayOfPedSquare[i_chamber][i_layer - 1][strip - 1] += pedMean*pedMean ;
		 
		 arrayPeak[i_chamber][i_layer-1][strip-1] = max-pedMean;
		 arrayOfPeak[i_chamber][i_layer - 1][strip - 1] += max-pedMean;
		 arrayOfPeakSquare[i_chamber][i_layer - 1][strip - 1] += (max-pedMean)*(max-pedMean);
		 
		 arraySumFive[i_chamber][i_layer-1][strip-1] = (max1-pedMean)/(max-pedMean);
		 
	       }//end digis loop
	     }//end if cfeb.available loop
	   }//end layer loop
	 }//end chamber loop
	 
	 eventNumber++;
	 edm::LogInfo ("CSCPedestalAnalyzer")  << "end of event number " << eventNumber;
       }
     }
   }
 
  
}


 

//define this as a plug-in
DEFINE_FWK_MODULE(CSCPedestalAnalyzer)
  
