

#include "L1Trigger/TextToDigi/src/TextToRaw.h"

// system
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/Exception.h"

// Raw data collection
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::ios;

const int TextToRaw::EVT_MAX_SIZE;

TextToRaw::TextToRaw(const edm::ParameterSet& iConfig) :
  filename_(iConfig.getUntrackedParameter<string>("filename", "slinkOutput.txt")),
  fedId_(iConfig.getUntrackedParameter<int>("fedId", 745))
{
  edm::LogInfo("TextToDigi") << "Reading ASCII dump from " << filename_ << endl;

  //register the products
  produces<FEDRawDataCollection>();

}


TextToRaw::~TextToRaw()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to produce the data  ------------
void
TextToRaw::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // read file
   string line;
   int i=0; // count 32-bit words

   // while not encountering dumb errors
   while (getline(file_, line) && !line.empty() ) {

     // convert string to int
     std::istringstream iss(line);
     unsigned long d;
     iss >> std::hex >> d;

     // copy data
     for (int j=0; j<4; j++) {
       if ( (i*4+j) < EVT_MAX_SIZE ) { 
	 char c = (d>>(8*j))&0xff;
	 data_[i*4+j] = c;
       }
     }

     ++i;

     // bail if we reached the EVT_MAX_SIZE
     if (i>=EVT_MAX_SIZE) {
       throw cms::Exception("TextToRaw")
	 << "Read too many lines from file. Maximum event size is " << EVT_MAX_SIZE << " lines" << std::endl;
     }

   }

   int evtSize = i * 4;

   // create the collection
   std::auto_ptr<FEDRawDataCollection> rawColl(new FEDRawDataCollection()); 
   // retrieve the target buffer
   FEDRawData& feddata=rawColl->FEDData(fedId_);
   // Allocate space for header+trailer+payload
   feddata.resize(evtSize);

   // fill FEDRawData object
   for (unsigned i=0; i<evtSize; ++i) {
     feddata.data()[i] = data_[i];
   }

   // put the collection in the event
   iEvent.put(rawColl);
     
}


// ------------ method called once each job just before starting event loop  ------------
void 
TextToRaw::beginJob(const edm::EventSetup&)
{
  // open VME file
  file_.open(filename_.c_str(), ios::in);
  if(!file_.good()) { edm::LogInfo("TextToDigi") << "Failed to open ASCII file " << filename_ << endl; }
}


// ------------ method called once each job just after ending the event loop  ------------
void 
TextToRaw::endJob() {
  file_.close();
}

