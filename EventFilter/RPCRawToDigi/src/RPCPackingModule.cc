#include "EventFilter/RPCRawToDigi/interface/RPCPackingModule.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/DataRecord/interface/RPCReadOutMappingRcd.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRawDataPacker.h"


using namespace std;
using namespace edm;


RPCPackingModule::RPCPackingModule( const ParameterSet& pset ) :
//  digiLabel_(""),
  eventCounter_(0)
{

  // Set some private data members
//  digiLabel_ = pset.getParameter<InputTag>("DigiProducer");

  // Define EDProduct type
  produces<FEDRawDataCollection>();

}

RPCPackingModule::~RPCPackingModule()
{}




void RPCPackingModule::produce( edm::Event& ev,
                              const edm::EventSetup& es)
{
  eventCounter_++;
  LogInfo("RPCPackingModule") << "[RPCPackingModule::produce] " 
                              << "event counter: " << eventCounter_;

  Handle< RPCDigiCollection > digiCollection;
  ev.getByType(digiCollection);

  ESHandle<RPCReadOutMapping> readoutMapping;
  es.get<RPCReadOutMappingRcd>().get(readoutMapping);

  auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );

  pair<int,int> rpcFEDS=FEDNumbering::getRPCFEDIds();
  for (int id= rpcFEDS.first; id<=rpcFEDS.second; ++id){

    RPCRecordFormatter formatter(id, readoutMapping.product()) ;

    FEDRawData *  rawData =  RPCRawDataPacker().rawData(id, digiCollection.product(), formatter);
//    FEDRawData *  rawData =  formatter.packData(id, digiCollection.product());
    FEDRawData& fedRawData = buffers->FEDData(id);

    fedRawData = *rawData;
  }
   

  ev.put( buffers );  
}

