

#include "EventFilter/GctRawToDigi/src/GctVmeToRaw.h"

// system
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Raw data collection
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::ios;


GctVmeToRaw::GctVmeToRaw(const edm::ParameterSet& iConfig) :
  filename_(iConfig.getUntrackedParameter<string>("filename", "slinkOutput.txt")),
  evtSize_(iConfig.getUntrackedParameter<int>("eventSize", 1024)),
  fedId_(iConfig.getUntrackedParameter<int>("fedId", 745))
{
  edm::LogInfo("GCT") << "Reading VME data from " << filename_ << endl;

  // open VME file
  file_.open(filename_.c_str(), ios::in);
  if(!file_.good()) { edm::LogInfo("GCT") << "Failed to open VME file " << filename_ << endl; }

  //register the products
  produces<FEDRawDataCollection>();

}


GctVmeToRaw::~GctVmeToRaw()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to produce the data  ------------
void
GctVmeToRaw::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // create the collection
   std::auto_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection()); 
   // retrieve the target buffer
   FEDRawData& feddata=rawColl->FEDData(fedId_);
   // Allocate space for header+trailer+payload
   feddata.resize(evtSize_);


   // read file
   string line;
   int i=0;
         
   while (file_ >> line && line!="" && i<evtSize_/4) {

     // convert string to int
     std::istringstream iss(line);
     unsigned long d;
     iss >> std::hex >> d;

     // copy data
     for (int j=0; j<4; j++) {
       if ( (i*4+j) < evtSize_ ) { 
	 char c = (d>>(8*j))&0xff;
	 feddata.data()[i*4+j] = c;
       }
     }

     ++i;
   }

   // put the collection in the event
   iEvent.put(rawColl);
     
}


// ------------ method called once each job just before starting event loop  ------------
void 
GctVmeToRaw::beginJob(const edm::EventSetup&)
{
}


// ------------ method called once each job just after ending the event loop  ------------
void 
GctVmeToRaw::endJob() {
}

