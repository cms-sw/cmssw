#include "EventFilter/SiPixelRawToDigi/interface/SiPixelRawToDigiModule.h"
#include "EventFilter/SiPixelRawToDigi/src/PixelDataFormatter.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"

#include "DataFormats/DetId/interface/DetId.h"

//#include "CalibTracker/SiPixelConnectivity/interface/PixelFEDConnectivity.h"
//#include "CalibTracker/SiPixelConnectivity/interface/PixelFEDConnections.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"


// -----------------------------------------------------------------------------
SiPixelRawToDigiModule::SiPixelRawToDigiModule( const edm::ParameterSet& conf ) :
  formatter(0), connectivity(0)
{
  produces<PixelDigiCollection>();
}


// -----------------------------------------------------------------------------
SiPixelRawToDigiModule::~SiPixelRawToDigiModule() {
//  delete formatter;
//  delete connectivity;
}


// -----------------------------------------------------------------------------
void SiPixelRawToDigiModule::beginJob(const edm::EventSetup& c) 
{
//  formatter = new PixelDataFormatter;
//  connectivity = new PixelFEDConnectivity;
}

// -----------------------------------------------------------------------------
void SiPixelRawToDigiModule::endJob() 
{

}

// -----------------------------------------------------------------------------
void SiPixelRawToDigiModule::produce( edm::Event& ev,
                              const edm::EventSetup& sp) 
{
  edm::Handle<FEDRawDataCollection> rawdata;
  ev.getByLabel("PixelDaqRawData", rawdata);

  // create producti (digis)
  std::auto_ptr<PixelDigiCollection> digis( new PixelDigiCollection );

  for (int id= 0; id<=FEDNumbering::lastFEDId(); ++id){

    const FEDRawData& data = rawdata->FEDData(id);
//    PixelFEDConnections * connections = connectivity.fed(id);
    if (data.size()){
      //formatter->interpretRawData(connections, data,  *digis );
    }
  }

  ev.put( digis );
}

