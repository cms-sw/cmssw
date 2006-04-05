using namespace std;

#include "EventFilter/SiPixelRawToDigi/interface/SiPixelDigiToRaw.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"


#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CalibTracker/SiPixelConnectivity/interface/SiPixelFedCablingMapBuilder.h"

#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"



#include <iostream>

SiPixelDigiToRaw::SiPixelDigiToRaw( const edm::ParameterSet& pset ) :
  eventCounter_(0),
  verbosity_(0),
  productLabel_(""),
  fedCablingMap_(0)
{
  std::cout << " HERE ** SiPixelDigiToRaw::SiPixelDigiToRaw]"
          << " Constructing object..." << std::endl;

  // Set some private data members
  productLabel_ = pset.getParameter<std::string>("DigiProducer");
  verbosity_    = pset.getParameter<int>("Verbosity");

  // Define EDProduct type
  produces<FEDRawDataCollection>();

}

// -----------------------------------------------------------------------------
SiPixelDigiToRaw::~SiPixelDigiToRaw() {
//  delete formatter;
//  delete connectivity;
  cout << " HERE ** SiPixelDigiToRaw destructor!, events: "
       <<eventCounter_ << endl;
}

// -----------------------------------------------------------------------------
void SiPixelDigiToRaw::beginJob(const edm::EventSetup& setup)
{
  cout << " HERE ** SiPixelDigiToRaw beginJob" << endl;

}

// -----------------------------------------------------------------------------
void SiPixelDigiToRaw::produce( edm::Event& ev,
                              const edm::EventSetup& es)
{
  eventCounter_++;
  if (verbosity_>0) std::cout << "[SiPixelDigiToRaw::produce] "
                        << "event number: "
                        << eventCounter_ << std::endl;


  PixelDataFormatter formatter;

  edm::Handle<PixelDigiCollection> digiCollection;
  ev.getByLabel(productLabel_, digiCollection);

  //FIXME - temporary solution
  typedef std::vector<unsigned int> DetIDs;
  DetIDs detIDs = digiCollection->detIDs();
  PixelDataFormatter::Digis digis;
  for( DetIDs::const_iterator it=detIDs.begin(); it != detIDs.end(); it++) {
   PixelDigiCollection::Range range = digiCollection->get(*it);
   digis[*it] =  std::vector<PixelDigi>(range.first,range.second); 
  }
  

  if( !fedCablingMap_) {
    fedCablingMap_ = SiPixelFedCablingMapBuilder().produce(es); 
  }

//  edm::ESHandle<SiPixelFedCabling> cabling;
//  es.get<SiPixelFedCablingRcd>().get( cabling );
//  cabling->myprintout();
  

  // create product (raw data)
  std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );

  
  vector<PixelFEDCabling *> cabling = fedCablingMap_->cabling();

  typedef vector<PixelFEDCabling *>::iterator FI;
  for (FI it = cabling.begin(); it != cabling.end(); it++) {
    cout << " PRODUCE DATA FOR FED_id: " << (**it).id() << endl;
    FEDRawData * rawData = formatter.formatData( (**it), digis);
    FEDRawData& fedRawData = buffers->FEDData( (**it).id() ); 
    cout << "** SiPixelDigiToRaw data size: " << sizeof(*rawData) << endl;
    fedRawData = *rawData;
    cout << "size of data in fedRawData: " << fedRawData.size() << endl;
  }
  

  ev.put( buffers );
  if (verbosity_>0) std::cout << "[SiPixelDigiToRaw::produce], after "<<endl;
  
}

// -----------------------------------------------------------------------------

