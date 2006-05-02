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
#include "CalibMuon/CSCCalibration/interface/CSCCrossTalkAnalyzer.h"


CSCCrossTalkAnalyzer::CSCCrossTalkAnalyzer(edm::ParameterSet const& conf) {
  
  eventNumber=0,evt=0;
  strip=0,misMatch=0;
  i_chamber=0,i_layer=0,reportedChambers=0;
  length=1;
  pedMean=0.0,time=0.0;

  for (int i=0;i<480;i++){
    new_xtalk_intercept_right[i] = -999.;
    new_xtalk_intercept_left[i]  = -999.;
    new_xtalk_slope_right[i]     = -999.;
    new_xtalk_slope_left[i]      = -999.;
    new_rchi2[i]                 = -999.;
    new_lchi2[i]                 = -999.;
    newPeakTime[i]               = -999.;
    newMeanPeakTime[i]           = -999.;
  }
  
  for (int i=0; i<CHAMBERS; i++){
    for (int j=0; j<LAYERS; j++){
      for (int k=0; k<STRIPS; k++){
        for (int l=0; l<TIMEBINS*20; l++){
          thetime[i][j][k][l]       = 0.0;
          thebins[i][j][k][l]       = 0;
          theadccountsc[i][j][k][l] = 0;
          theadccountsl[i][j][k][l] = 0;
          theadccountsr[i][j][k][l] = 0;
        }
      }
    }
  }
  
  for (int i=0; i<CHAMBERS; i++){
    for (int j=0; j<LAYERS; j++){
      for (int k=0; k<STRIPS; k++){
        xtalk_intercept_left[i][j][k]  = -999.;
        xtalk_intercept_right[i][j][k] = -999.;
        xtalk_slope_left[i][j][k]      = -999.;
        xtalk_slope_right[i][j][k]     = -999.;
        xtalk_chi2_left[i][j][k]       = -999.;
        xtalk_chi2_right[i][j][k]      = -999.;
        myPeakTime[i][j][k]            = 0.0;
        myMeanPeakTime[i][j][k]        = 0.0;
        array_meanPeakTime[i][j][k]    = -999.;
      }
    }
  }  
}

void CSCCrossTalkAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
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
      
      evt++;      
      for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) {  ///loop over DDUs
	
	///get a reference to chamber data
	const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	
	reportedChambers += dduData[iDDU].header().ncsc();
	int NChambers = cscData.size();
	int repChambers = dduData[iDDU].header().ncsc();
	std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	if (NChambers!=repChambers) { std::cout<< "misMatched size!!!" << std::endl; misMatch++;}

	for (int chamber = 0; chamber < NChambers; chamber++){
	  
	  for (int layer = 1; layer <= 6; layer++){
	    
	    std::vector<CSCStripDigi> digis = cscData[chamber].stripDigis(layer) ;
	    const CSCDMBHeader &thisDMBheader = cscData[chamber].dmbHeader();
	    
            if (thisDMBheader.cfebAvailable()){
              dmbID[chamber] = cscData[chamber].dmbHeader().dmbID();//get DMB ID
              crateID[chamber] = cscData[chamber].dmbHeader().crateID();//get crate ID
              if(crateID[chamber] == 255) continue;
	      
              for (unsigned int i=0; i<digis.size(); i++){
                int strip = digis[i].getStrip();
                std::vector<int> adc = digis[i].getADCCounts();
		
                int offset = evt / 20;
                int smain[5],splus[5],sminus[5]; //5 for CFEBs
                for(int s=0;s<5;s++) smain[s]  = s*16+offset;
                for(int s=0;s<5;s++) splus[s]  = s*16+offset+1;
                for(int s=0;s<5;s++) sminus[s] = s*16+offset-1;
                int iuse=-99;
                for(int s=0; s<5; s++) {if(strip-1==smain[s])  iuse=smain[s];}
                for(int s=0; s<5; s++) {if(strip-1==splus[s])  iuse=smain[s];}
                for(int s=0; s<5; s++) {if(strip-1==sminus[s]) iuse=smain[s];}
		
                if(iuse!=-99){
		  
                  for(unsigned int k=0;k<adc.size();k++){
		    
                    time = (50. * k)-((evt%20)* 6.25)+116.5;
                    pedMean =(adc[0]+adc[1])/2;
		    
                    int kk=8*k-evt%20+19;//19 to zero everything, for binning 120
		    
                    thebins[chamber][layer-1][strip-1][kk] = 8*k-evt%20+19;
		    thetime[chamber][layer-1][strip-1][kk] = time;
		    
                    if(iuse==strip-1)  theadccountsc[chamber][layer-1][iuse][kk] = adc[k];
                    if(iuse==strip)    theadccountsr[chamber][layer-1][iuse][kk] = adc[k];
                    if(iuse==strip-2)  theadccountsl[chamber][layer-1][iuse][kk] = adc[k];
		    
                  }//end loop over timebins
                }//end iuse!=99
              }//end loop over digis
            }//end cfeb.available loop
          }//end loop over layers
        }//end loop over chambers

	eventNumber++;
	edm::LogInfo ("CSCCrossTalkAnalyzer")  << "end of event number " << eventNumber;

      }
    }
  }
}



//define this as a plug-in
DEFINE_FWK_MODULE(CSCCrossTalkAnalyzer)
  
