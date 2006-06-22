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
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "OnlineDB/CSCCondDB/interface/CSCscaAnalyzer.h"

//bool CSCscaAnalyzer::debug=false;

CSCscaAnalyzer::CSCscaAnalyzer(edm::ParameterSet const& conf) {
  debug = conf.getUntrackedParameter<bool>("debug",false);
  eventNumber=0,evt=0, Nddu=0;
  strip=0,misMatch=0;
  chamber=0,layer=0,reportedChambers=0;
  length=1,myevt=0,flag=-9;
  pedMean=0.0,NChambers=0;

  //definition of histograms
}

void CSCscaAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
 
  edm::Handle<CSCStripDigiCollection> strips;
  e.getByLabel("cscunpacker","MuonCSCStripDigi",strips);
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByType(rawdata); //before 0_7_0_pre4 use getByLabel("DaqSource", rawdata)
  //myevt=e.id().event();
  
  for (int id=FEDNumbering::getCSCFEDIds().first;
       id<=FEDNumbering::getCSCFEDIds().second; ++id){ //for each of our DCCs
    
    /// Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);
    if (fedData.size()){ ///unpack data 
      
      ///get a pointer to data and pass it to constructor for unpacking
      CSCDCCEventData dccData((short unsigned int *) fedData.data()); 
      
      const std::vector<CSCDDUEventData> & dduData = dccData.dduData(); 
      
      evt++;      
      for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) { 
	
	///get a reference to chamber data
	const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	Nddu = dduData.size();
	reportedChambers += dduData[iDDU].header().ncsc();
	NChambers = cscData.size();
	int repChambers = dduData[iDDU].header().ncsc();
	std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	if (NChambers!=repChambers) { std::cout<< "misMatched size!!!" << std::endl; misMatch++; continue;}
	

	for (int chamber = 0; chamber < NChambers; chamber++){
	  const CSCDMBHeader &thisDMBheader = cscData[chamber].dmbHeader();
	  
	  if (thisDMBheader.cfebAvailable()){
	    dmbID[chamber] = cscData[chamber].dmbHeader().dmbID();
	    crateID[chamber] = cscData[chamber].dmbHeader().crateID();
	    if(crateID[chamber] == 255) continue;

	    for (int icfeb=0; icfeb<5;icfeb++) {//loop over cfebs in a given chamber
	      CSCCFEBData * mycfebData =  cscData[chamber].cfebData(icfeb);
	      if (!mycfebData) continue;

	      for (unsigned int layer = 1; layer <= 6; layer++){
		std::vector<CSCStripDigi> digis = cscData[chamber].stripDigis(layer) ;		
	
		//	for (unsigned int i=0; i<digis.size(); i++){//digis size
		//int strip = digis[i].getStrip();
		//size[chamber] = digis.size();
		  for (int itime=0;itime<8;itime++){
		    CSCCFEBTimeSlice * mytimeSlice =  mycfebData->timeSlice(itime);
		    if (!mytimeSlice)continue;
		    
		    scaBlock = mytimeSlice->scaControllerWord(layer).sca_blk;
		    trigTime = mytimeSlice->scaControllerWord(layer).trig_time;
		    lctPhase = mytimeSlice->scaControllerWord(layer).lct_phase;
		    int tmp=1;
		    for(power=0;power<8;power++){if(trigTime==tmp) lctPhase=power; tmp=tmp*2;}
		    cap = lctPhase+digis.size();
		    scaNumber=8*scaBlock+cap;
		    //std::vector<int> adc = digis[i].getADCCounts();
		    std::cout <<" Cap "<<cap<<" CFEB "<<icfeb<<" Layer "<<layer<<" sca_block "<<scaBlock <<" trig_time "<<trigTime<<" lct_phase "<<lctPhase<<" sca_number "<<scaNumber<<" timeslice "<<itime<< std::endl; 
		    
		  }//timeslice
		  //}//digis size
	      }//layer
	    }//CFEBs
	  }//CFEB available
	}//chamber
	
	eventNumber++;
	edm::LogInfo ("CSCscaAnalyzer")  << "end of event number " << eventNumber;
	
      }
    }
  }
}
