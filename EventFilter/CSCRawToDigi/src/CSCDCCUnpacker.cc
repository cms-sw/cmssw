#include "EventFilter/CSCRawToDigi/interface/CSCDCCUnpacker.h"

//Framework stuff
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

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


  instatiateDQM = pset.getUntrackedParameter<bool>("runDQM", false);
  if(instatiateDQM){
   
   monitor = edm::Service<CSCMonitorInterface>().operator->(); 
  
  }
  

  //std::cout << "starting DCCConstructor";   

  //fill constructor here
  //dccData = 0;
  produces<CSCWireDigiCollection>();
  produces<CSCStripDigiCollection>();
  //std::cout <<"... and finished " << std::endl;  
   
  CSCAnodeData::setDebug(true);
  CSCALCTHeader::setDebug(true);
  CSCCLCTData::setDebug(true);
  CSCEventData::setDebug(true);
  CSCTMBData::setDebug(true);
  CSCDCCEventData::setDebug(true);
  CSCDDUEventData::setDebug(true);
  CSCTMBHeader::setDebug(true);
  //move this outside of producer
  std::string releasetop(getenv("CMSSW_BASE"));
  std::string mappingFilePath= releasetop + "/src/CondFormats/CSCObjects/test/";    
  std::string mappingFileName = mappingFilePath + "csc_slice_test_map.txt";
  theMapping  = CSCReadoutMappingFromFile( mappingFileName );
  
}

CSCDCCUnpacker::~CSCDCCUnpacker(){
  
  //fill destructor here
  //delete dccData;   

}


void CSCDCCUnpacker::produce(edm::Event & e, const edm::EventSetup& c){

 


  // Get a handle to the FED data collection
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByLabel("DaqRawData", rawdata);

  // create the collection of CSC wire and strip Digis
  std::auto_ptr<CSCWireDigiCollection> wireProduct(new CSCWireDigiCollection);
  std::auto_ptr<CSCStripDigiCollection> stripProduct(new CSCStripDigiCollection);
  //std::auto_ptr<CSCALCTDigiCollection> alctProduct(new CSCALCTDigiCollection);
  //std::auto_ptr<CSCCLCTpDigiCollection> clctProduct(new CSCCLCTDigiCollection);
  //std::auto_ptr<CSCComparatorDigiCollection> ComparatorProduct(new CSCComparatorDigiCollection);
  //std::auto_ptr<CSCRPCDigiCollection> RPCProduct(new CSCRPCDigiCollection);
  
  //std::cout <<"in the producer now " << std::endl;  

  for (int id=FEDNumbering::getCSCFEDIds().first;
       id<=FEDNumbering::getCSCFEDIds().second; ++id){ //for each of our DCCs

    
  //for (int id=1873;id<=1873; ++id){ //for each of our DCCs


  //std::cout <<"in the loop of CSCFEDs now " << std::endl;
    
    // Take a reference to this FED's data
    const FEDRawData& fedData = rawdata->FEDData(id);


    if (fedData.size()){ //unpack data 
      //std::cout <<"in the loop of CSCFEDs data now " << std::endl;
     
      //get a pointer to data and pass it to constructor for unpacking
      CSCDCCEventData dccData((short unsigned int *) fedData.data()); 
      
      numOfEvents++; 
     

      if(instatiateDQM) monitor->process(dccData);

      std::cout<<"**************[DCCUnpackingModule]:"<<numOfEvents<<" events analyzed"<<std::endl;



      //get a reference to dduData
      const std::vector<CSCDDUEventData> & dduData = dccData.dduData(); 

      for (int iDDU=0; iDDU<dduData.size(); ++iDDU) {  //loop over DDUs
	
	//get a reference to chamber data
	const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();
	
	for (int iCSC=0; iCSC<cscData.size(); ++iCSC) { //loop over CSCs

	  ///Digis for each chamber must be obtained here	
	  ///below is an example for  wire digis 
	  ///it must be repeated for all 6 types!
	  
	  for (int ilayer = 1; ilayer <= 6; ilayer++) { 
	    int endcap = 1;
	    int station = 1;
	    int tmb = 1;
 	    int vmecrate = cscData[iCSC].dmbHeader().crateID(); 
	    int dmb = cscData[iCSC].dmbHeader().dmbID();

	    std::cout << "crate = " << vmecrate << "; dmb = " << dmb << std::endl;           

	    CSCDetId layer(1, //endcap
			   1, //station
			   1, //ring
			   1, //chamber
			   ilayer); //layer
 
            if (((vmecrate==0)||(vmecrate==1)) && (dmb>=0)&&(dmb<=10)&&(dmb!=6)) {
	      layer = theMapping.detId( endcap, station, vmecrate, dmb, tmb,ilayer );
	    }else {
	      std::cerr << " detID input out of range!!! " << std::endl;
	      std::cerr << " using fake CSCDetId!!!! " << std::endl;
	    }

	    std::vector <CSCWireDigi> wireDigis =  cscData[iCSC].wireDigis(ilayer);
	    for (int i=0; i<wireDigis.size() ; i++) {
	      wireProduct->insertDigi(layer, wireDigis[i]);
	    }

	    std::vector <CSCStripDigi> stripDigis =  cscData[iCSC].stripDigis(ilayer);
            for (int i=0; i<stripDigis.size() ; i++) {
              stripProduct->insertDigi(layer, stripDigis[i]);
            }

	  }
	}
      }     
    }
  }
  // commit to the event
  e.put(wireProduct);
  e.put(stripProduct);
  //e.put(ALCTProduct);
  //e.put(CLCTProduct);
  //e.put(ComparatorProduct);
  //e.put(RPCProduct);

}




