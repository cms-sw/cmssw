/** \file
 *
 *  $Date: 2008/01/22 18:58:23 $
 *  $Revision: 1.6 $
 *  \author A. Tumanov - Rice
 */

#include <EventFilter/CSCRawToDigi/src/CSCDigiToRawModule.h>
#include <EventFilter/CSCRawToDigi/src/CSCDigiToRaw.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/CSCChamberMapRcd.h"



using namespace edm;
using namespace std;

CSCDigiToRawModule::CSCDigiToRawModule(const edm::ParameterSet & pset): 
  packer(new CSCDigiToRaw),
  theStripDigiTag(pset.getParameter<edm::InputTag>("stripDigiTag")),
  theWireDigiTag(pset.getParameter<edm::InputTag>("wireDigiTag"))
{
  //theStrip = pset.getUntrackedParameter<string>("DigiCreator", "cscunpacker");
  produces<FEDRawDataCollection>("CSCRawData"); 
}


CSCDigiToRawModule::~CSCDigiToRawModule(){
  delete packer;
}


void CSCDigiToRawModule::produce(Event & e, const EventSetup& c){
  ///reverse mapping for packer
  edm::ESHandle<CSCChamberMap> hcham;
  c.get<CSCChamberMapRcd>().get(hcham); 
  const CSCChamberMap* theMapping = hcham.product();


  auto_ptr<FEDRawDataCollection> fed_buffers(new FEDRawDataCollection);
  // Take digis from the event
  Handle<CSCStripDigiCollection> stripDigis;
  e.getByLabel(theStripDigiTag, stripDigis);
  Handle<CSCWireDigiCollection> wireDigis;
  e.getByLabel(theWireDigiTag, wireDigis);


  // Create the packed data
  packer->createFedBuffers(*stripDigis, *wireDigis, *(fed_buffers.get()), theMapping, e);


  
  // put the raw data to the event
  e.put(fed_buffers, "CSCRawData");
}


