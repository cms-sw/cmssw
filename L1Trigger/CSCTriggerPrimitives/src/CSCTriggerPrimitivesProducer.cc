//-------------------------------------------------
//
//   Class: CSCTriggerPrimitivesProducer
//
//   Description: Steering routine of the local Level-1 Cathode Strip Chamber
//                trigger.
//
//   Author List: S. Valuev (May 2006)
//
//   Modifications:
//
//--------------------------------------------------
 
#include <L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesProducer.h>
#include <L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesBuilder.h>

#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>

#include <DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>

CSCTriggerPrimitivesProducer::CSCTriggerPrimitivesProducer(const edm::ParameterSet& conf) : iev(0) {

  wireDigiProducer_ = conf.getParameter<std::string>("CSCWireDigiProducer");
  compDigiProducer_ = conf.getParameter<std::string>("CSCComparatorDigiProducer");

  lctBuilder_ = new CSCTriggerPrimitivesBuilder(conf); // pass on the conf

  // register what this produces
  produces<CSCALCTDigiCollection>();
  produces<CSCCLCTDigiCollection>();
  produces<CSCCorrelatedLCTDigiCollection>();
}

CSCTriggerPrimitivesProducer::~CSCTriggerPrimitivesProducer() {
  LogDebug("L1CSCTrigger")
    << "deleting trigger primitives after " << iev << " events.";
  delete lctBuilder_;
}

void CSCTriggerPrimitivesProducer::produce(edm::Event& ev,
					   const edm::EventSetup& setup) {
  LogDebug("L1CSCTrigger") << "start producing LCTs for event " << ++iev;

  // Find the geometry (& conditions?) for this event & cache it in 
  // CSCTriggerGeometry.
  edm::ESHandle<CSCGeometry> h;
  setup.get<MuonGeometryRecord>().get(h);
  CSCTriggerGeometry::setGeometry(h);

  // Get the collections of comparator & wire digis from event.
  edm::Handle<CSCComparatorDigiCollection> compDigis;
  edm::Handle<CSCWireDigiCollection>       wireDigis;
  ev.getByLabel(compDigiProducer_, "MuonCSCComparatorDigi", compDigis);
  ev.getByLabel(wireDigiProducer_, "MuonCSCWireDigi",       wireDigis);

  // Create empty collections of ALCTs, CLCTs, and correlated LCTs.
  std::auto_ptr<CSCALCTDigiCollection> oc_alct(new CSCALCTDigiCollection);
  std::auto_ptr<CSCCLCTDigiCollection> oc_clct(new CSCCLCTDigiCollection);
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> oc_lct(new CSCCorrelatedLCTDigiCollection);

  // Fill collections.
  lctBuilder_->build(wireDigis.product(), compDigis.product(),
		     *oc_alct, *oc_clct, *oc_lct);

  // Put collections in event.
  ev.put(oc_alct);
  ev.put(oc_clct);
  ev.put(oc_lct);
}
