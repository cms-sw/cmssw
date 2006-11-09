/** \file
 *
 *  $Date: 2006/11/09 11:35:25 $
 *  $Revision: 1.0 $
 *  \author A. Tumanov - Rice
 */

#include <EventFilter/CSCRawToDigi/src/CSCDigiToRawModule.h>
#include <EventFilter/CSCRawToDigi/src/CSCDigiToRaw.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

using namespace edm;
using namespace std;

CSCDigiToRawModule::CSCDigiToRawModule(const edm::ParameterSet & pset) : packer(new CSCDigiToRaw) {}


CSCDigiToRawModule::~CSCDigiToRawModule(){
  delete packer;
}


void CSCDigiToRawModule::produce(Event & e, const EventSetup& c){

  auto_ptr<FEDRawDataCollection> fed_buffers(new FEDRawDataCollection);

  // Take digis from the event
  Handle<CSCStripDigiCollection> digis;
  e.getByLabel("CSCDigis", digis);

  // Create the packed data
  packer->createFedBuffers(*digis, *(fed_buffers.get()));
  
  // put the raw data to the event
  e.put(fed_buffers);
}


