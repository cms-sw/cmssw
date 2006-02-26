#include "EventFilter/CSCTFRawToDigi/interface/CSCTFUnpacker.h"

//Framework stuff
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

//FEDRawData 
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

//Digi stuff
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
//#include "DataFormats/CSCTFObjects/interface/CSCTFL1Track.h"

//Include LCT digis later
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>
//#include "DataFormats/CSCTFObjects/interface/CSCTFL1TrackCollection.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include <EventFilter/CSCTFRawToDigi/interface/CSCTFMonitorInterface.h>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/CSCObjects/interface/CSCTriggerMappingFromFile.h"
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

//CSC Track Finder Raw Data Formats // TB and DDU
#include "TBDataFormats/CSCTFTBRawData/interface/CSCTFTBEventData.h"
//#include "DataFormats/CSCTFRawData/interface/CSCTFEventData.h"
// more to come later

#include <iostream>


CSCTFUnpacker::CSCTFUnpacker(const edm::ParameterSet & pset)
{

  instantiateDQM = pset.getUntrackedParameter<bool>("runDQM", false);
  testBeam = pset.getUntrackedParameter<bool>("TestBeamData",false);
  std::string mapPath = pset.getUntrackedParameter<std::string>("MappingFile","");
  if(testBeam) 
    {
      TBFEDid = pset.getUntrackedParameter<int>("TBFedId");
      TBendcap = pset.getUntrackedParameter<int>("TBEndcap");
      TBsector = pset.getUntrackedParameter<int>("TBSector");
    }
  else
    {
      TBFEDid = 0;
      TBsector = 0;
      TBendcap = 0;
    }
  debug = pset.getUntrackedParameter<bool>("debug",false);
  TFmapping = new CSCTriggerMappingFromFile(getenv("CMSSW_BASE")+mapPath);

  if(instantiateDQM){
   
   monitor = edm::Service<CSCTFMonitorInterface>().operator->(); 
  
  }  

  if(debug) std::cout << "starting CSCTFConstructor";   

  produces<CSCCorrelatedLCTDigiCollection>();
  //produces<CSCTFL1TrackCollection>();

  if(debug) std::cout <<"... and finished " << std::endl;  
}

CSCTFUnpacker::~CSCTFUnpacker(){
  
  //fill destructor here
  //delete dccData;   
  delete TFmapping;
}


void CSCTFUnpacker::produce(edm::Event & e, const edm::EventSetup& c)
{
  
  //create data pointers, to use later
  // CSCTFDDUEventData * ddu = NULL;
  CSCTFTBEventData *tbdata = NULL;
  
  // Get a handle to the FED data collection
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByLabel("DaqRawData", rawdata);
  
  // create the collection of CSC wire and strip Digis
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> LCTProduct(new CSCCorrelatedLCTDigiCollection);
  //std::auto_ptr<CSCTFL1TrackCollection> trackProduct(new CSCRPCDigiCollection);
  
  std::cout <<"in the producer now " << std::endl;  
  
  for(int fedid = ((testBeam) ? (TBFEDid) : FEDNumbering::getCSCFEDIds().first);
      fedid <= ((testBeam) ? (TBFEDid) : FEDNumbering::getCSCFEDIds().second);
      ++fedid)
  {
    std::cout<<"In TF FED Id (loop, maybe?)\n";
    const FEDRawData& fedData = rawdata->FEDData(fedid);
    if(fedData.size())
      {
	std::cout<<"Starting Data Unpack\n";
	if(testBeam) 
	  tbdata = new CSCTFTBEventData(reinterpret_cast<unsigned short*>(fedData.data()));
	else
	  std::cout << "not implemented yet, waiting on hardware\n";
	
	++numOfEvents;
	
	if(instantiateDQM)
	  { 
	    if(tbdata) monitor->process(*tbdata);
	    else std::cout<<"not implemented yet\n";
	  }

	std::cout<<"CSCTFUnpacker::produce "<< numOfEvents << " events unpacked.\n";

	CSCTFTBFrontBlock aFB;
	CSCTFTBSPBlock aSPB;	
	CSCTFTBSPData aSPD;
	for(int BX = 1; BX<=7 ; ++BX)
	  {
	    if(testBeam) aFB = tbdata->frontDatum(BX);
	    for(int FPGA = 1; FPGA <=5 ; ++FPGA)
	      {		
		  for(int MPClink = 1; MPClink <= 3 ; ++MPClink)
		    {		      
		      if(testBeam)
		      {		    
			int subsector = 0;
			int station = 0;
			
			if(FPGA == 1) subsector = 1;
			if(FPGA == 2) subsector = 2;
			station = (((FPGA - 1) == 0) ? 1 : FPGA-1);
			
			int cscid = aFB.frontData(FPGA,MPClink).CSCIDPacked();
			if(cscid)
			  {			    
			    CSCDetId id = TFmapping->detId(TBendcap,station,TBsector,subsector,cscid);
			    LCTProduct->insertDigi(id,aFB.frontDigiData(FPGA,MPClink));
			  }
			
		      }
		      else 
		      {
			std::cout<<"not implemented yet\n";
		      }
		    }
	      }
	  }
      }
  }

  e.put(LCTProduct); // put processed lcts into the event.

  if(tbdata) delete tbdata;
  tbdata = NULL;
}




