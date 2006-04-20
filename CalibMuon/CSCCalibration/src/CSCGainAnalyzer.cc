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
#include "CalibMuon/CSCCalibration/interface/CSCGainAnalyzer.h"

CSCGainAnalyzer::CSCGainAnalyzer(edm::ParameterSet const& conf) {
  
  eventNumber=0,evt=0;
  strip=0,misMatch=0;
  i_chamber=0,i_layer=0,reportedChambers=0;
  length=1,gainSlope=-999.0,gainIntercept=-999.0;
  
  for (int i=0; i<NUMMODTEN; i++){
    for (int j=0; j<CHAMBERS; j++){
      for (int k=0; k<LAYERS; k++){
	for (int l=0;l<STRIPS;l++){
	  maxmodten[i][j][k][l] = 0.0;
	}
      }
    }
  }
  
  for (int i=0; i<CHAMBERS; i++){
    for (int j=0; j<LAYERS; j++){
      for (int k=0; k<STRIPS; k++){
	arrayOfGain[i][j][k]       = -999.0;
	arrayOfGainSquare[i][j][k] = -999.0;
	arrayOfGain[i][j][k]       = -999.0;
	arrayOfIntercept[i][j][k]  = -999.0;
	arrayOfInterceptSquare[i][j][k]=-999.0;
	arrayOfChi2[i][j][k]       = -999.0;
	adcMax[i][j][k]            = -999.0;
	adcMean_max[i][j][k]       = -999.0;
      }
    }
  }
  
  for (int i=0; i<480; i++){
    newGain[i]     =0.0;
    newIntercept[i]=0.0;
    newChi2[i]     =0.0;
  }
}

void CSCGainAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
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
	int NChambers = cscData.size();
	int repChambers = dduData[iDDU].header().ncsc();
	std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	if (NChambers!=repChambers) { std::cout<< "misMatched size!!!" << std::endl; misMatch++;}
	
	evt++;

	for (int i_chamber=0; i_chamber<NChambers; i_chamber++) {   
	  
	  for (int i_layer = 1; i_layer <=6; ++i_layer) {
	    
	    std::vector<CSCStripDigi> digis = cscData[i_chamber].stripDigis(i_layer) ;
	    const CSCDMBHeader &thisDMBheader = cscData[i_chamber].dmbHeader();
	    
	    if (thisDMBheader.cfebAvailable()){
	      dmbID[i_chamber] = cscData[i_chamber].dmbHeader().dmbID();//get DMB ID
	      crateID[i_chamber] = cscData[i_chamber].dmbHeader().crateID();//get crate ID
	      if(crateID[i_chamber] == 255) continue;
	      
	      for (unsigned int i=0; i<digis.size(); i++){
		std::vector<int> adc = digis[i].getADCCounts();
		strip = digis[i].getStrip();
		adcMax[i_chamber][i_layer-1][strip-1]=-99.0; 
		for(unsigned int k=0;k<adc.size();k++){
		  float ped=(adc[0]+adc[1])/2.;
		  if(adc[k]-ped > adcMax[i_chamber][i_layer-1][strip-1]) {
		    adcMax[i_chamber][i_layer-1][strip-1]=adc[k]-ped;
		  }
		}
		adcMean_max[i_chamber][i_layer-1][strip-1] += adcMax[i_chamber][i_layer-1][strip-1]/20.;  
		
		//On the 10th event save
		if (evt%20 == 0 && (strip-1)%16 == (evt-1)/200){
		  int ten = int((evt-1)/20)%10 ;
		  maxmodten[ten][i_chamber][i_layer-1][strip-1] = adcMean_max[i_chamber][i_layer-1][strip-1];
		}
	      }//end digis loop
	    }//end cfeb.available loop
	  }//end layer loop
	}//end chamber loop
	
	if((evt-1)%20==0){
	  for(int ii=0; ii<CHAMBERS; ii++){
	    for(int jj=0; jj<LAYERS; jj++){
	      for(int kk=0; kk<STRIPS; kk++){
		adcMean_max[ii][jj][kk]=0.0;
	      }
	    }
	  }
	}
	eventNumber++;
	edm::LogInfo ("CSCGainAnalyzer")  << "end of event number " << eventNumber;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCGainAnalyzer)
