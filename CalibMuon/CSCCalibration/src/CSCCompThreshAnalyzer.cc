/** 
 * Analyzer for reading CSC comapartor thresholds.
 * author O.Boeriu 9/05/06 
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
#include "CalibMuon/CSCCalibration/interface/CSCCompThreshAnalyzer.h"

CSCCompThreshAnalyzer::CSCCompThreshAnalyzer(edm::ParameterSet const& conf) {

  eventNumber = 0;
  evt = 0,Nddu=0,misMatch=0,event=0;
  i_chamber=0,i_layer=0,reportedChambers =0;
  length = 1, NChambers=0;
  timebin=-999,mycompstrip=-999,comparator=0,compstrip=0;

  for(int i=0;i<CHAMBERS;i++){
    for(int j=0; j<LAYERS; j++){
      for(int k=0; k<STRIPS; k++){
	theMeanThresh[i][j][k] = 0.;
	arrayMeanThresh[i][j][k] = 0.;
	mean[i][j][k]=0.;
	meanTot[i][j][k]=0.;
      }
    }
  }

  for (int i=0;i<480;i++){
    newThresh[i]=0;
  }
  
}

void CSCCompThreshAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  
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
	 if (NChambers!=repChambers) {std::cout<< "misMatched size!!!" << std::endl; misMatch++;continue;}
	 	 
	 for (i_chamber=0; i_chamber<NChambers; i_chamber++) {//loop over all DMBs  
	   if (cscData[i_chamber].nclct()) {
	     CSCCLCTData & clctData = cscData[i_chamber].clctData();
	   }else {
	     std::cout<<" No CLCT!" <<std::endl;
	     continue;
	   }
	   CSCCLCTData & clctData = cscData[i_chamber].clctData();
	   for(i_layer = 1; i_layer <= 6; ++i_layer) {//loop over all layers in chambers
	     //std::vector<CSCStripDigi> digis = cscData[i_chamber].stripDigis(i_layer) ;
	     std::vector<CSCComparatorDigi> comp = clctData.comparatorDigis(i_layer);
	     
	     const CSCDMBHeader &thisDMBheader = cscData[i_chamber].dmbHeader();
	     
	     if (thisDMBheader.cfebAvailable()){//check that CFEB data exists
	       
	       dmbID[i_chamber]   = cscData[i_chamber].dmbHeader().dmbID(); //get DMB ID
	       crateID[i_chamber] = cscData[i_chamber].dmbHeader().crateID(); //get crate ID
	       if(crateID[i_chamber] == 255) continue; //255 is reserved for old crate, present only 0 and 1
	       
	       for (unsigned int i=0; i<comp.size(); i++){//loop over CFEB comparator digis
		 comparator = comp[i].getComparator();
		 timebin = comp[i].getTimeBin() ;
		 compstrip =  comp[i].getStrip();
		 int this_comparator[4] = {4, 5, 6, 7};
		 
		 for (int iii=0; iii<40; iii++){
		   if ((compstrip == iii) && (comparator == this_comparator[0] || comparator == this_comparator[1])) {
		     mycompstrip = 0 + iii*2;
		   } else if ((compstrip == iii) && (comparator == this_comparator[2] || comparator == this_comparator[3])) {
		     mycompstrip = 1 + iii*2;
		   }
		 }
		 
		 mean[i_chamber][i_layer-1][mycompstrip] = comparator/5;
		 
	       }//end comp loop
	       
	       meanTot[i_chamber][i_layer-1][mycompstrip] +=mean[i_chamber][i_layer-1][mycompstrip]/25.;
	       
	       // On the 25th event
	       if (evt%25 == 0&&(mycompstrip)%16==(evt-1)/875){
		 int tmp = int((evt-1)/25)%35 ;
		 meanmod[tmp][i_chamber][i_layer-1][mycompstrip] = meanTot[i_chamber][i_layer-1][mycompstrip];
	       }
	     }//end if cfeb.available loop
	   }//end layer loop
	 }//end chamber loop

	 if((evt-1)%25==0){
	   for(int ii=0;ii<CHAMBERS;ii++){
	     for(int jj=0;jj<LAYERS;jj++){
	       for(int kk=0;kk<STRIPS;kk++){
		 mean[ii][jj][kk]=0.0;
		 meanTot[ii][jj][kk]=0.0;
	       }
	     }
	   }
	 }
	 
	 eventNumber++;
	 edm::LogInfo ("CSCCompThreshAnalyzer")  << "end of event number " << eventNumber;
	 
       }//end DDU loop
     }
   }
}
