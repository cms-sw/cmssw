using namespace std;
#include "EventFilter/SiPixelRawToDigi/interface/SiPixelRawToDigi.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"


#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CalibTracker/SiPixelConnectivity/interface/SiPixelFedCablingMapBuilder.h"

#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"


#include <iostream>


// -----------------------------------------------------------------------------
SiPixelRawToDigi::SiPixelRawToDigi( const edm::ParameterSet& conf ) 
  : eventCounter_(0), verbosity_(0),  fedCablingMap_(0)
{
  cout << " HERE ** SiPixelRawToDigi constructor!" << endl;
  verbosity_ =  conf.getParameter<int>("Verbosity");
  cout << " Verbosity is: " << verbosity_ << endl;
  produces<PixelDigiCollection>();
}


// -----------------------------------------------------------------------------
SiPixelRawToDigi::~SiPixelRawToDigi() {
//  delete formatter;
//  delete connectivity;
  cout << " HERE ** SiPixelRawToDigi destructor!" << endl;
}


// -----------------------------------------------------------------------------
void SiPixelRawToDigi::beginJob(const edm::EventSetup& c) 
{
  cout << " HERE ** SiPixelRawToDigi beginJob" << endl;
//  formatter = new PixelDataFormatter;
//  connectivity = new PixelFEDConnectivity;
}

// -----------------------------------------------------------------------------
void SiPixelRawToDigi::produce( edm::Event& ev,
                              const edm::EventSetup& es) 
{
  PixelDataFormatter formatter;

  if( !fedCablingMap_) {
    fedCablingMap_ = SiPixelFedCablingMapBuilder().produce(es);
  }


  edm::Handle<FEDRawDataCollection> buffers;
//  ev.getByLabel("PixelDaqRawData", rawdata);
  ev.getByType(buffers);

  // create producti (digis)
  std::auto_ptr<PixelDigiCollection> collection( new PixelDigiCollection );

  vector<PixelFEDCabling *> cabling = fedCablingMap_->cabling();
  typedef vector<PixelFEDCabling *>::iterator FI;
  for (FI it = cabling.begin(); it != cabling.end(); it++) {
     PixelDataFormatter::Digis digis;
     cout << " PRODUCE DIGI FOR FED: " <<  (**it).id() << endl;
     
     const FEDRawData& fedRawData = buffers->FEDData( (**it).id() );
     cout << "sizeof data buffer: " << fedRawData.size() << endl;
     formatter.interpretRawData( **it, fedRawData, digis);

     typedef PixelDataFormatter::Digis::iterator ID;
     for (ID it = digis.begin(); it != digis.end(); it++) {
       uint32_t detid = it->first;
       PixelDigiCollection::ContainerIterator beg = it->second.begin(); 
       PixelDigiCollection::ContainerIterator end = it->second.end(); 
       collection->put( PixelDigiCollection::Range(beg,end), detid);
     } 
  }

  ev.put( collection );
}

