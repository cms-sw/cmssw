/** \file
 *  \author A. Tumanov - Rice
 */

#include "EventFilter/CSCRawToDigi/src/CSCDigiToRawModule.h"
#include "EventFilter/CSCRawToDigi/src/CSCDigiToRaw.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/CSCChamberMapRcd.h"

CSCDigiToRawModule::CSCDigiToRawModule(const edm::ParameterSet & pset): 
  packer(new CSCDigiToRaw(pset))
{
  //theStrip = pset.getUntrackedParameter<string>("DigiCreator", "cscunpacker");
  
  theFormatVersion =  pset.getUntrackedParameter<unsigned int>("useFormatVersion", 2005); 	// pre-LS1 - '2005'. post-LS1 - '2013'
  usePreTriggers = pset.getUntrackedParameter<bool>("usePreTriggers", true); 			// disable checking CLCT PreTriggers digis

  wd_token = consumes<CSCWireDigiCollection>( pset.getParameter<edm::InputTag>("wireDigiTag") );
  sd_token = consumes<CSCStripDigiCollection>( pset.getParameter<edm::InputTag>("stripDigiTag") );
  cd_token = consumes<CSCComparatorDigiCollection>( pset.getParameter<edm::InputTag>("comparatorDigiTag") );
  pr_token = consumes<CSCCLCTPreTriggerCollection>( pset.getParameter<edm::InputTag>("preTriggerTag") );
  al_token = consumes<CSCALCTDigiCollection>( pset.getParameter<edm::InputTag>("alctDigiTag") );
  cl_token = consumes<CSCCLCTDigiCollection>( pset.getParameter<edm::InputTag>("clctDigiTag") );
  co_token = consumes<CSCCorrelatedLCTDigiCollection>( pset.getParameter<edm::InputTag>("correlatedLCTDigiTag") );

  produces<FEDRawDataCollection>("CSCRawData"); 

}


CSCDigiToRawModule::~CSCDigiToRawModule(){
  delete packer;
}


void CSCDigiToRawModule::produce( edm::Event & e, const edm::EventSetup& c ){
  ///reverse mapping for packer
  edm::ESHandle<CSCChamberMap> hcham;
  c.get<CSCChamberMapRcd>().get(hcham); 
  const CSCChamberMap* theMapping = hcham.product();

  std::auto_ptr<FEDRawDataCollection> fed_buffers(new FEDRawDataCollection);

  // Take digis from the event
  edm::Handle<CSCWireDigiCollection> wireDigis;
  edm::Handle<CSCStripDigiCollection> stripDigis;
  edm::Handle<CSCComparatorDigiCollection> comparatorDigis;
  edm::Handle<CSCALCTDigiCollection> alctDigis;
  edm::Handle<CSCCLCTDigiCollection> clctDigis;
  edm::Handle<CSCCLCTPreTriggerCollection> preTriggers;
  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedLCTDigis;

  e.getByToken( wd_token, wireDigis );
  e.getByToken( sd_token, stripDigis );
  e.getByToken( cd_token, comparatorDigis );
  e.getByToken( al_token, alctDigis );
  e.getByToken( cl_token, clctDigis );
  if (usePreTriggers)
     e.getByToken( pr_token, preTriggers );
  e.getByToken( co_token, correlatedLCTDigis );

  // Create the packed data
  packer->createFedBuffers(*stripDigis, *wireDigis, *comparatorDigis, 
                           *alctDigis, *clctDigis, *preTriggers, *correlatedLCTDigis,
                           *(fed_buffers.get()), theMapping, e, theFormatVersion, usePreTriggers);
  
  // put the raw data to the event
  e.put(fed_buffers, "CSCRawData");
}


