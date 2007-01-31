

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
  evtSize_(iConfig.getUntrackedParameter<int>("eventSize", 512))
{
  edm::LogInfo("GCT") << "GctVmeToRaw : reading VME data from " << filename_ << endl;

  // open VME file
  file_.open(filename_.c_str(), ios::in);
  if(!file_.good()) { edm::LogInfo("GCT") << "GctVmeToRaw : could not open " << filename_ << endl; }

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

   // create the FEDRawData
   FEDRawData rawEvt(evtSize_);

   // read file
   string line;
   int i=0;
         
   while (file_ >> line && line!="") {

     // convert string to int
     std::istringstream iss(line);
     unsigned long d;
     iss >> std::hex >> d;

     cout << std::hex << d << endl;

     // copy to raw data
     if (rawEvt.size() < (i+1)*4) { rawEvt.resize( (i*4)+8 ); }
     for (int j=0; j<4; j++) {
       if ( i*4+j < rawEvt.size()) {
	 //cout << std::hex << ((d>>(8*j))&0xff) << endl;
	 rawEvt.data()[i*4+j] = (d>>(8*j))&0xff;
       }
       else {
	 cerr << "VME data bigger the FEDRawData container : "  << rawEvt.size() << endl;
       }
     }

     i++;

   }

   rawColl->FEDData(1) = rawEvt;
   
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

