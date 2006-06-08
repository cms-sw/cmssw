/** 
 * Analyzer for reading AFEB thresholds.
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

#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCLCTData.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "IORawData/CSCCommissioning/src/FileReaderDDU.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "CalibMuon/CSCCalibration/interface/CSCAFEBdacAnalyzer.h"

CSCAFEBdacAnalyzer::CSCAFEBdacAnalyzer(edm::ParameterSet const& conf) {

  i_chamber=0,i_layer=0,reportedChambers =0;
  eventNumber = 0,event=0;
  evt = 0,Nddu=0;
  length = 1, NChambers=0;
  misMatch=0, wireGroup=0, wireTBin=0;

  
}



void CSCAFEBdacAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
   edm::Handle<CSCWireDigiCollection> wires;
   
  // Pass the handle to the method "getByType", which is used to retrieve
  // one and only one instance of the type in question out of event "e". If
  // zero or more than one instance exists in the event an exception is thrown.
  //

   e.getByLabel("cscunpacker","MuonCSCWireDigi",wires);
  
   edm::Handle<FEDRawDataCollection> rawdata;
   e.getByLabel("DaqSource" , rawdata);
   event =e.id().event();
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
	 Nddu=dduData.size();
	 reportedChambers += dduData[iDDU].header().ncsc();
	 NChambers = cscData.size();
	 int repChambers = dduData[iDDU].header().ncsc();
	 std::cout << " Reported Chambers = " << repChambers <<"   "<<NChambers<< std::endl;
	 if (NChambers!=repChambers) { std::cout<< "misMatched size!!!" << std::endl; misMatch++;continue;}

	 for (i_chamber=0; i_chamber<NChambers; i_chamber++) {//loop over all DMBs
	   
	   for(i_layer = 1; i_layer <= 6; ++i_layer) {//loop over all layers in chambers
	     //int wire[64]=0;
	     
	     std::vector<CSCWireDigi> wire = cscData[i_chamber].wireDigis(i_layer) ;
	     const CSCDMBHeader &thisDMBheader = cscData[i_chamber].dmbHeader();
	     //if (thisDMBheader.afebAvailable()){};//check that AFEB data exists
	     
	     dmbID[i_chamber]   = cscData[i_chamber].dmbHeader().dmbID(); 
	     crateID[i_chamber] = cscData[i_chamber].dmbHeader().crateID(); 
	     if(crateID[i_chamber] == 255) continue;
	     
	     for (unsigned int i=0; i < wire.size(); i++){ //loop over wire digis
	       wireGroup = wire[i].getWireGroup();
	       wireTBin = wire[i].getBeamCrossingTag();wireTBin = wire[i].getBeamCrossingTag();
	       
	       
	     }//end digis loop
	   }//end layer loop
	 }//end chamber loop
	 
	 eventNumber++;
	 edm::LogInfo ("CSCAFEBdacAnalyzer")  << "end of event number " << eventNumber;
       }//end DDU loop
     }
   }
}
