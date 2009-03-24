//-------------------------------------------------
//
//   Class: CSCTriggerPrimitivesProducer
//
//   Description: Steering routine of the local Level-1 Cathode Strip Chamber
//                trigger.
//
//   Author List: S. Valuev, UCLA.
//
//   $Date: 2008/10/09 11:09:22 $
//   $Revision: 1.6 $
//
//   Modifications:
//
//--------------------------------------------------
 
#include <L1Trigger/CSCTriggerPrimitives/plugins/CSCTriggerPrimitivesProducer.h>
#include <L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesBuilder.h>

#include <Utilities/Timing/interface/TimingReport.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>

#include <DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>

// Configuration via EventSetup
#include "CondFormats/CSCObjects/interface/CSCL1TPParameters.h"
#include "CondFormats/DataRecord/interface/CSCL1TPParametersRcd.h"


CSCTriggerPrimitivesProducer::CSCTriggerPrimitivesProducer(const edm::ParameterSet& conf) : iev(0) {

  wireDigiProducer_ = conf.getParameter<edm::InputTag>("CSCWireDigiProducer");
  compDigiProducer_ = conf.getParameter<edm::InputTag>("CSCComparatorDigiProducer");

  lctBuilder_ = new CSCTriggerPrimitivesBuilder(conf); // pass on the conf

  // register what this produces
  produces<CSCALCTDigiCollection>();
  produces<CSCCLCTDigiCollection>();
  produces<CSCCorrelatedLCTDigiCollection>();
  produces<CSCCorrelatedLCTDigiCollection>("MPCSORTED");
}

CSCTriggerPrimitivesProducer::~CSCTriggerPrimitivesProducer() {
  LogDebug("L1CSCTrigger")
    << "deleting trigger primitives after " << iev << " events.";
  delete lctBuilder_;
  //TimingReport::current()->dump(std::cout);
}

//void CSCTriggerPrimitivesProducer::beginRun(const edm::EventSetup& setup) {
//}

void CSCTriggerPrimitivesProducer::produce(edm::Event& ev,
					   const edm::EventSetup& setup) {
  //static TimingReport::Item & lctTimer =
  //  (*TimingReport::current())["CSCTriggerPrimitivesProducer:produce"];
  //TimeMe t(lctTimer, false);

  LogDebug("L1CSCTrigger") << "start producing LCTs for event " << ++iev;

  // Find the geometry (& conditions?) for this event & cache it in 
  // CSCTriggerGeometry.
  edm::ESHandle<CSCGeometry> h;
  setup.get<MuonGeometryRecord>().get(h);
  CSCTriggerGeometry::setGeometry(h);

  // Get config. parameters using EventSetup mechanism.  This must be done
  // in produce() for every event and not in beginJob() (see mail from
  // Jim Brooke sent to hn-cms-L1TrigEmulator on July 30, 2007).
  edm::ESHandle<CSCL1TPParameters> conf;
  setup.get<CSCL1TPParametersRcd>().get(conf);
  if (conf.product() == 0) {
    edm::LogError("CSCTriggerPrimitivesProducer")
      << "+++ Failed to find a CSCL1TPParametersRcd in EventSetup! +++\n"
      << "+++ Cannot continue emulation without these parameters +++\n";
    return;
  }
  lctBuilder_->setConfigParameters(conf.product());

  // Get the collections of comparator & wire digis from event.
  edm::Handle<CSCComparatorDigiCollection> compDigis;
  edm::Handle<CSCWireDigiCollection>       wireDigis;
  ev.getByLabel(compDigiProducer_.label(), compDigiProducer_.instance(), compDigis);
  ev.getByLabel(wireDigiProducer_.label(), wireDigiProducer_.instance(), wireDigis);

  // Create empty collections of ALCTs, CLCTs, and correlated LCTs upstream
  // and downstream of MPC.
  std::auto_ptr<CSCALCTDigiCollection> oc_alct(new CSCALCTDigiCollection);
  std::auto_ptr<CSCCLCTDigiCollection> oc_clct(new CSCCLCTDigiCollection);
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> oc_lct(new CSCCorrelatedLCTDigiCollection);
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> oc_sorted_lct(new CSCCorrelatedLCTDigiCollection);

  // Fill collections.
  lctBuilder_->build(wireDigis.product(), compDigis.product(),
		     *oc_alct, *oc_clct, *oc_lct, *oc_sorted_lct);

  // Put collections in event.
  ev.put(oc_alct);
  ev.put(oc_clct);
  ev.put(oc_lct);
  ev.put(oc_sorted_lct,"MPCSORTED");
}
