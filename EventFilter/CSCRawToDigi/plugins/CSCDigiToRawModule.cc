/** \file
 *
 *  $Date: 2010/07/20 02:58:29 $
 *  $Revision: 1.12 $
 *  \author A. Tumanov - Rice
 */

#include "EventFilter/CSCRawToDigi/src/CSCDigiToRawModule.h"
#include "EventFilter/CSCRawToDigi/src/CSCDigiToRaw.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/CSCChamberMapRcd.h"



using namespace edm;
using namespace std;

CSCDigiToRawModule::CSCDigiToRawModule(const edm::ParameterSet & pset): 
  packer(new CSCDigiToRaw(pset)),
  theStripDigiTag(pset.getParameter<edm::InputTag>("stripDigiTag")),
  theWireDigiTag(pset.getParameter<edm::InputTag>("wireDigiTag")),
  theComparatorDigiTag(pset.getParameter<edm::InputTag>("comparatorDigiTag")),
  theALCTDigiTag(pset.getParameter<edm::InputTag>("alctDigiTag")),
  theCLCTDigiTag(pset.getParameter<edm::InputTag>("clctDigiTag")),
  thePreTriggerTag(pset.getParameter<edm::InputTag>("preTriggerTag")),
  theCorrelatedLCTDigiTag(pset.getParameter<edm::InputTag>("correlatedLCTDigiTag"))
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
  Handle<CSCComparatorDigiCollection> comparatorDigis;
  e.getByLabel(theComparatorDigiTag, comparatorDigis);
  Handle<CSCALCTDigiCollection> alctDigis;
  e.getByLabel(theALCTDigiTag, alctDigis);
  Handle<CSCCLCTDigiCollection> clctDigis;
  e.getByLabel(theCLCTDigiTag, clctDigis);
  Handle<CSCCLCTPreTriggerCollection> preTriggers;
  e.getByLabel(thePreTriggerTag, preTriggers);
  Handle<CSCCorrelatedLCTDigiCollection> correlatedLCTDigis;
  e.getByLabel(theCorrelatedLCTDigiTag, correlatedLCTDigis);



  // Create the packed data
  packer->createFedBuffers(*stripDigis, *wireDigis, *comparatorDigis, 
                           *alctDigis, *clctDigis, *preTriggers, *correlatedLCTDigis,
                           *(fed_buffers.get()), theMapping, e);
  
  // put the raw data to the event
  e.put(fed_buffers, "CSCRawData");
}


