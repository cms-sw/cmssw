/** 
 * Analyzer for reading CSC bin by bin ADC information.
 * author S. Durkin, O.Boeriu 26/04/06 
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
#include "CalibMuon/CSCCalibration/interface/AutoCorrMat.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixAnalyzer.h"

CSCNoiseMatrixAnalyzer::CSCNoiseMatrixAnalyzer(edm::ParameterSet const& conf) {
  
  eventNumber=0,evt=0,NChambers=0,Nddu=0;
  strip=0,misMatch=0;
  i_chamber=0,i_layer=0,reportedChambers=0;
  length=1;
  for(int k=0;k<CHAMBERS;k++) cam[k].zero();

  for (int i=0;i<480;i++){
    newMatrix1[i] = 0.0;
    newMatrix2[i] = 0.0;
    newMatrix3[i] = 0.0;
    newMatrix4[i] = 0.0;
    newMatrix5[i] = 0.0;
    newMatrix6[i] = 0.0;
    newMatrix7[i] = 0.0;
    newMatrix8[i] = 0.0;
    newMatrix9[i] = 0.0;
    newMatrix10[i]= 0.0;
    newMatrix11[i]= 0.0;
    newMatrix12[i]= 0.0;
  
  }

  for (int i=0; i< CHAMBERS; i++){
    size[i]=0;
  }
}


void CSCNoiseMatrixAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
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
    
    evt++;      
    /// Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);
    if (fedData.size()){ ///unpack data 
      
      ///get a pointer to data and pass it to constructor for unpacking
      CSCDCCEventData dccData((short unsigned int *) fedData.data()); 
      
      const std::vector<CSCDDUEventData> & dduData = dccData.dduData(); 
         
      for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) {  ///loop over DDUs
	
	///get a reference to chamber data
	const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	Nddu=dduData.size();
	reportedChambers += dduData[iDDU].header().ncsc();
	NChambers = cscData.size();
	int repChambers = dduData[iDDU].header().ncsc();
	std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	if (NChambers!=repChambers) { std::cout<< "misMatched size!!!" << std::endl; misMatch++;}

	for (int i_chamber=0; i_chamber<NChambers; i_chamber++) { 
	  
	  for(int i_layer = 1; i_layer <= LAYERS; ++i_layer) {
	    std::vector<CSCStripDigi> digis = cscData[i_chamber].stripDigis(i_layer) ;
	    const CSCDMBHeader &thisDMBheader = cscData[i_chamber].dmbHeader();
	    
	    if (thisDMBheader.cfebAvailable()){
	      dmbID[i_chamber]   = cscData[i_chamber].dmbHeader().dmbID(); 
	      crateID[i_chamber] = cscData[i_chamber].dmbHeader().crateID();
	      if(crateID[i_chamber] == 255) continue; 

	      for (unsigned int i=0; i<digis.size(); i++){
		size[i_chamber]=digis.size();
		int strip = digis[i].getStrip();
		adc = digis[i].getADCCounts();
		int tadc[8];
		for(unsigned int j=0;j<adc.size();j++)tadc[j]=adc[j];
		cam[i_chamber].add(i_layer-1,strip-1,tadc);
	      }
	    }
	  }
	}
	tmp=corrmat; 
		
	eventNumber++;
	edm::LogInfo ("CSCNoiseMatrixAnalyzer")  << "end of event number " << eventNumber;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCNoiseMatrixAnalyzer)
    
