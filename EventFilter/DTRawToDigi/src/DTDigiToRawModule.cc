/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include <EventFilter/DTRawToDigi/src/DTDigiToRawModule.h>
#include <EventFilter/DTRawToDigi/src/DTDigiToRaw.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

using namespace edm;
using namespace std;

DTDigiToRawModule::DTDigiToRawModule() : packer(new DTDigiToRaw) {}


DTDigiToRawModule::~DTDigiToRawModule(){
  delete packer;
}


void DTDigiToRawModule::produce(Event & e, const EventSetup& c){

  auto_ptr<FEDRawDataCollection> fed_buffers(new FEDRawDataCollection);

  // Take digis from the event
  Handle<DTDigiCollection> digis;
  e.getByLabel("DTDigis", digis);

  // Create the packed data
  packer->createFedBuffers(*digis, *(fed_buffers.get()));
  
  // put the raw data to the event
  e.put(fed_buffers);
}

