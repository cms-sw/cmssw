//-------------------------------------------------
//
//   Class: CSCTriggerPrimitivesProducer
//
//   Description: Steering routine of the local Level-1 Cathode Strip Chamber
//                trigger.
//
//   Author List: S. Valuev, UCLA.
//
//   $Id: CSCTriggerPrimitivesProducer.cc,v 1.15.2.1 2012/05/16 00:31:23 khotilov Exp $
//
//   Modifications:
//
//--------------------------------------------------
 
#include "L1Trigger/CSCTriggerPrimitives/plugins/CSCTriggerPrimitivesProducer.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesBuilder.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"

#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

// Configuration via EventSetup
#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"
#include "CondFormats/DataRecord/interface/CSCDBL1TPParametersRcd.h"


CSCTriggerPrimitivesProducer::CSCTriggerPrimitivesProducer(const edm::ParameterSet& conf) : iev(0) {

  // if false, parameters will be read in from DB using EventSetup mechanism
  // else will use all parameters from the config file
  debugParameters_ = conf.getUntrackedParameter<bool>("debugParameters",false);

  wireDigiProducer_ = conf.getParameter<edm::InputTag>("CSCWireDigiProducer");
  compDigiProducer_ = conf.getParameter<edm::InputTag>("CSCComparatorDigiProducer");
  checkBadChambers_ = conf.getUntrackedParameter<bool>("checkBadChambers", true);

  lctBuilder_ = new CSCTriggerPrimitivesBuilder(conf); // pass on the conf

  // register what this produces
  produces<CSCALCTDigiCollection>();
  produces<CSCCLCTDigiCollection>();
  produces<CSCCLCTPreTriggerCollection>();
  produces<CSCCorrelatedLCTDigiCollection>();
  produces<CSCCorrelatedLCTDigiCollection>("MPCSORTED");
}

CSCTriggerPrimitivesProducer::~CSCTriggerPrimitivesProducer() {
  LogDebug("L1CSCTrigger")
    << "deleting trigger primitives after " << iev << " events.";
  delete lctBuilder_;
}

//void CSCTriggerPrimitivesProducer::beginRun(const edm::EventSetup& setup) {
//}

void CSCTriggerPrimitivesProducer::produce(edm::Event& ev,
					   const edm::EventSetup& setup) {

  LogDebug("L1CSCTrigger") << "start producing LCTs for event " << ++iev;

  // Find the geometry (& conditions?) for this event & cache it in 
  // CSCTriggerGeometry.
  {
    edm::ESHandle<CSCGeometry> h;
    setup.get<MuonGeometryRecord>().get(h);
    CSCTriggerGeometry::setGeometry(h);
  }

  // Find conditions data for bad chambers.
  edm::ESHandle<CSCBadChambers> pBadChambers;
  setup.get<CSCBadChambersRcd>().get(pBadChambers);

  // If !debugParameters then get config parameters using EventSetup mechanism.
  // This must be done in produce() for every event and not in beginJob() 
  // (see mail from Jim Brooke sent to hn-cms-L1TrigEmulator on July 30, 2007).
  if (!debugParameters_) {
    edm::ESHandle<CSCDBL1TPParameters> conf;
    setup.get<CSCDBL1TPParametersRcd>().get(conf);
    if (conf.product() == 0) {
      edm::LogError("L1CSCTPEmulatorConfigError")
        << "+++ Failed to find a CSCDBL1TPParametersRcd in EventSetup! +++\n"
        << "+++ Cannot continue emulation without these parameters +++\n";
      return;
    }
    lctBuilder_->setConfigParameters(conf.product());
  }
  
  // Get the collections of comparator & wire digis from event.
  edm::Handle<CSCComparatorDigiCollection> compDigis;
  edm::Handle<CSCWireDigiCollection>       wireDigis;
  ev.getByLabel(compDigiProducer_.label(), compDigiProducer_.instance(), compDigis);
  ev.getByLabel(wireDigiProducer_.label(), wireDigiProducer_.instance(), wireDigis);

  // Create empty collections of ALCTs, CLCTs, and correlated LCTs upstream
  // and downstream of MPC.
  std::auto_ptr<CSCALCTDigiCollection> oc_alct(new CSCALCTDigiCollection);
  std::auto_ptr<CSCCLCTDigiCollection> oc_clct(new CSCCLCTDigiCollection);
  std::auto_ptr<CSCCLCTPreTriggerCollection> oc_pretrig(new CSCCLCTPreTriggerCollection);
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> oc_lct(new CSCCorrelatedLCTDigiCollection);
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> oc_sorted_lct(new CSCCorrelatedLCTDigiCollection);

  if (!wireDigis.isValid()) {
    edm::LogWarning("L1CSCTPEmulatorNoInputCollection")
      << "+++ Warning: Collection of wire digis with label "
      << wireDigiProducer_.label()
      << " requested in configuration, but not found in the event..."
      << " Skipping production of CSC TP digis +++\n";
  }
  if (!compDigis.isValid()) {
    edm::LogWarning("L1CSCTPEmulatorNoInputCollection")
      << "+++ Warning: Collection of comparator digis with label "
      << compDigiProducer_.label()
      << " requested in configuration, but not found in the event..."
      << " Skipping production of CSC TP digis +++\n";
  }
  // Fill output collections if valid input collections are available.
  if (wireDigis.isValid() && compDigis.isValid()) {   
    const CSCBadChambers* temp = checkBadChambers_ ? pBadChambers.product() : new CSCBadChambers;
    lctBuilder_->build(temp,
		       wireDigis.product(), compDigis.product(),
		       *oc_alct, *oc_clct, *oc_pretrig, *oc_lct, *oc_sorted_lct);
    if (!checkBadChambers_)
      delete temp;
  }

  // Put collections in event.
  ev.put(oc_alct);
  ev.put(oc_clct);
  ev.put(oc_pretrig);
  ev.put(oc_lct);
  ev.put(oc_sorted_lct,"MPCSORTED");
}
