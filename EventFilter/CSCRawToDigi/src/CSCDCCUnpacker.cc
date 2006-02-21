#include "EventFilter/CSCRawToDigi/interface/CSCDCCUnpacker.h"

//Framework stuff
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//FEDRawData 
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

//Digi stuff
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
//#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
//#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
//Include LCT digis later

#include "DataFormats/CSCDigi/interface/CSCRPCDigi.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
//#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
//#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCLCTData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCRPCData.h"

#include <EventFilter/CSCRawToDigi/interface/CSCMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h"

#include <iostream>


CSCDCCUnpacker::CSCDCCUnpacker(const edm::ParameterSet & pset) :
  numOfEvents(0){

  debug = pset.getUntrackedParameter<bool>("Debug", false);
  std::string mappingFileName = pset.getUntrackedParameter<std::string>("theMappingFile",
								  "csc_slice_test_map.txt");

  instatiateDQM = pset.getUntrackedParameter<bool>("runDQM", false);
  if(instatiateDQM){
   
   monitor = edm::Service<CSCMonitorInterface>().operator->(); 
  
  }
  

  produces<CSCWireDigiCollection>("MuonCSCWireDigi");
  produces<CSCStripDigiCollection>("MuonCSCStripDigi");
  produces<CSCComparatorDigiCollection>("MuonCSCComparatorDigi");  
 
  CSCAnodeData::setDebug(debug);
  CSCALCTHeader::setDebug(debug);
  CSCCLCTData::setDebug(debug);
  CSCEventData::setDebug(debug);
  CSCTMBData::setDebug(debug);
  CSCDCCEventData::setDebug(debug);
  CSCDDUEventData::setDebug(debug);
  CSCTMBHeader::setDebug(debug);
  
  theMapping  = CSCReadoutMappingFromFile( mappingFileName );
  
}

CSCDCCUnpacker::~CSCDCCUnpacker(){
  
  //fill destructor here

}


void CSCDCCUnpacker::produce(edm::Event & e, const edm::EventSetup& c){

 


  // Get a handle to the FED data collection
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByLabel("DaqSource" , rawdata);

  // create the collection of CSC wire and strip Digis
  std::auto_ptr<CSCWireDigiCollection> wireProduct(new CSCWireDigiCollection);
  std::auto_ptr<CSCStripDigiCollection> stripProduct(new CSCStripDigiCollection);
  //std::auto_ptr<CSCALCTDigiCollection> alctProduct(new CSCALCTDigiCollection);
  //std::auto_ptr<CSCCLCTpDigiCollection> clctProduct(new CSCCLCTDigiCollection);
  std::auto_ptr<CSCComparatorDigiCollection> comparatorProduct(new CSCComparatorDigiCollection);
  //std::auto_ptr<CSCRPCDigiCollection> RPCProduct(new CSCRPCDigiCollection);
  
 
  numOfEvents++;
  
  //this line is to skip unpacking until 1309th event
  //if (numOfEvents>1308) {

  for (int id=FEDNumbering::getCSCFEDIds().first;
       id<=FEDNumbering::getCSCFEDIds().second; ++id){ //for each of our DCCs

    
  //for (int id=1873;id<=1873; ++id){ //for each of our DCCs


    
    // Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);


    if (fedData.size()){ //unpack data 
     
      //get a pointer to data and pass it to constructor for unpacking
      CSCDCCEventData dccData((short unsigned int *) fedData.data()); 
      

      if(instatiateDQM) monitor->process(dccData);

      //get a reference to dduData
      const std::vector<CSCDDUEventData> & dduData = dccData.dduData(); 

      for (int iDDU=0; iDDU<dduData.size(); ++iDDU) {  //loop over DDUs
	
	//get a reference to chamber data
	const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	
	for (int iCSC=0; iCSC<cscData.size(); ++iCSC) { //loop over CSCs

	  //this loop stores strip and wire digis:
	  for (int ilayer = 1; ilayer <= 6; ilayer++) { 
	    int endcap = 1;
	    int station = 1;
	    int tmb = 1;
 	    int vmecrate = cscData[iCSC].dmbHeader().crateID(); 
	    int dmb = cscData[iCSC].dmbHeader().dmbID();

	    if (debug) 
	      edm::LogInfo ("CSCDCCUnpacker") << "crate = " << vmecrate << "; dmb = " << dmb;
 
	    CSCDetId layer(1, //endcap
			   1, //station
			   1, //ring
			   1, //chamber
			   ilayer); //layer
 
            if (((vmecrate==0)||(vmecrate==1)) && (dmb>=0)&&(dmb<=10)&&(dmb!=6)) {
	      layer = theMapping.detId( endcap, station, vmecrate, dmb, tmb,ilayer );
	    }else {
	      edm::LogError ("CSCDCCUnpacker") << " detID input out of range!!! ";
	      edm::LogError ("CSCDCCUnpacker") << " using fake CSCDetId!!!! ";
	    }

	    std::vector <CSCWireDigi> wireDigis =  cscData[iCSC].wireDigis(ilayer);
	    for (int i=0; i<wireDigis.size() ; i++) {
	      wireProduct->insertDigi(layer, wireDigis[i]);
	    }

	    std::vector <CSCStripDigi> stripDigis =  cscData[iCSC].stripDigis(ilayer);
            for (int i=0; i<stripDigis.size() ; i++) {
              stripProduct->insertDigi(layer, stripDigis[i]);
            }

            if (ilayer == 3) { 
	      int nclct = cscData[iCSC].dmbHeader().nclct();
	      if (nclct) {
		if (cscData[iCSC].clctData().check()) {
		  std::vector <CSCComparatorDigi> comparatorDigis =
		    cscData[iCSC].clctData().comparatorDigis(3);
		  for (int i=0; i<comparatorDigis.size() ; i++) {
		    comparatorProduct->insertDigi(layer, comparatorDigis[i]);
		  }
		}
	      }
	    }
	  }
	}  
      }     
    }
  }
  edm::LogInfo("CSCDCCUnpacker") <<"**************[DCCUnpackingModule]:"<< std::ios::dec << numOfEvents<<" events analyzed ";
  //}
  // commit to the event
  e.put(wireProduct,"MuonCSCWireDigi");
  e.put(stripProduct,"MuonCSCStripDigi");
  //e.put(ALCTProduct);
  //e.put(CLCTProduct);
  e.put(comparatorProduct,"MuonCSCComparatorDigi");
  //e.put(RPCProduct);

}




