//-------------------------------------------------
//
//   Class: CSCTriggerPrimitivesProducer
//
//   Description: Steering routine of the local Level-1 Cathode Strip Chamber
//                trigger.
//
//   Author List: S. Valuev, UCLA.
//
//
//   Modifications:
//
//--------------------------------------------------

#include "L1Trigger/CSCTriggerPrimitives/plugins/CSCTriggerPrimitivesProducer.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCTriggerPrimitivesBuilder.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTPreTriggerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"

// Configuration via EventSetup
#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"

CSCTriggerPrimitivesProducer::CSCTriggerPrimitivesProducer(const edm::ParameterSet& conf) {
  config_ = conf;

  // if false, parameters will be read in from DB using EventSetup mechanism
  // else will use all parameters from the config file
  debugParameters_ = conf.getParameter<bool>("debugParameters");

  wireDigiProducer_ = conf.getParameter<edm::InputTag>("CSCWireDigiProducer");
  compDigiProducer_ = conf.getParameter<edm::InputTag>("CSCComparatorDigiProducer");
  gemPadDigiClusterProducer_ = conf.getParameter<edm::InputTag>("GEMPadDigiClusterProducer");

  checkBadChambers_ = conf.getParameter<bool>("checkBadChambers");

  keepCLCTPreTriggers_ = conf.getParameter<bool>("keepCLCTPreTriggers");
  keepALCTPreTriggers_ = conf.getParameter<bool>("keepALCTPreTriggers");
  keepShowers_ = conf.getParameter<bool>("keepShowers");

  // check whether you need to run the integrated local triggers
  const edm::ParameterSet commonParam(conf.getParameter<edm::ParameterSet>("commonParam"));
  runME11ILT_ = commonParam.getParameter<bool>("runME11ILT");
  runME21ILT_ = commonParam.getParameter<bool>("runME21ILT");

  wire_token_ = consumes<CSCWireDigiCollection>(wireDigiProducer_);
  comp_token_ = consumes<CSCComparatorDigiCollection>(compDigiProducer_);
  if (runME11ILT_ or runME21ILT_)
    gem_pad_cluster_token_ = consumes<GEMPadDigiClusterCollection>(gemPadDigiClusterProducer_);
  cscToken_ = esConsumes<CSCGeometry, MuonGeometryRecord>();
  gemToken_ = esConsumes<GEMGeometry, MuonGeometryRecord>();
  pBadChambersToken_ = esConsumes<CSCBadChambers, CSCBadChambersRcd>();
  confToken_ = esConsumes<CSCDBL1TPParameters, CSCDBL1TPParametersRcd>();

  // register what this produces
  produces<CSCALCTDigiCollection>();
  produces<CSCCLCTDigiCollection>();
  produces<CSCCLCTPreTriggerCollection>();
  if (keepCLCTPreTriggers_) {
    produces<CSCCLCTPreTriggerDigiCollection>();
  }
  if (keepALCTPreTriggers_) {
    produces<CSCALCTPreTriggerDigiCollection>();
  }
  produces<CSCCorrelatedLCTDigiCollection>();
  produces<CSCCorrelatedLCTDigiCollection>("MPCSORTED");
  if (keepShowers_) {
    produces<CSCShowerDigiCollection>();
    produces<CSCShowerDigiCollection>("Anode");
  }
  if (runME11ILT_ or runME21ILT_) {
    produces<GEMCoPadDigiCollection>();
  }
  // temporarily switch to a "one" module with a CSCTriggerPrimitivesBuilder data member
  builder_ = std::make_unique<CSCTriggerPrimitivesBuilder>(config_);
}

CSCTriggerPrimitivesProducer::~CSCTriggerPrimitivesProducer() {}

void CSCTriggerPrimitivesProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {
  // get the csc geometry
  builder_->setCSCGeometry(&setup.getData(cscToken_));

  // get the gem geometry if it's there
  edm::ESHandle<GEMGeometry> h_gem = setup.getHandle(gemToken_);
  if (h_gem.isValid()) {
    builder_->setGEMGeometry(&*h_gem);
  } else {
    edm::LogInfo("CSCTriggerPrimitivesProducer|NoGEMGeometry")
        << "+++ Info: GEM geometry is unavailable. Running CSC-only trigger algorithm. +++\n";
  }

  // Find conditions data for bad chambers.
  edm::ESHandle<CSCBadChambers> pBadChambers = setup.getHandle(pBadChambersToken_);

  // If !debugParameters then get config parameters using EventSetup mechanism.
  // This must be done in produce() for every event and not in beginJob()
  // (see mail from Jim Brooke sent to hn-cms-L1TrigEmulator on July 30, 2007).
  if (!debugParameters_) {
    edm::ESHandle<CSCDBL1TPParameters> conf = setup.getHandle(confToken_);
    if (conf.product() == nullptr) {
      edm::LogError("CSCTriggerPrimitivesProducer|ConfigError")
          << "+++ Failed to find a CSCDBL1TPParametersRcd in EventSetup! +++\n"
          << "+++ Cannot continue emulation without these parameters +++\n";
      return;
    }
    builder_->setConfigParameters(conf.product());
  }

  // Get the collections of comparator & wire digis from event.
  edm::Handle<CSCComparatorDigiCollection> compDigis;
  edm::Handle<CSCWireDigiCollection> wireDigis;
  ev.getByToken(comp_token_, compDigis);
  ev.getByToken(wire_token_, wireDigis);

  // input GEM pad cluster collection for upgrade scenarios
  const GEMPadDigiClusterCollection* gemPadClusters = nullptr;
  if (runME11ILT_ or runME21ILT_) {
    if (!gemPadDigiClusterProducer_.label().empty()) {
      edm::Handle<GEMPadDigiClusterCollection> gemPadDigiClusters;
      ev.getByToken(gem_pad_cluster_token_, gemPadDigiClusters);
      gemPadClusters = gemPadDigiClusters.product();
    }
  }

  // Create empty collections of ALCTs, CLCTs, and correlated LCTs upstream
  // and downstream of MPC.
  std::unique_ptr<CSCALCTDigiCollection> oc_alct(new CSCALCTDigiCollection);
  std::unique_ptr<CSCCLCTDigiCollection> oc_clct(new CSCCLCTDigiCollection);
  std::unique_ptr<CSCCLCTPreTriggerDigiCollection> oc_clctpretrigger(new CSCCLCTPreTriggerDigiCollection);
  std::unique_ptr<CSCALCTPreTriggerDigiCollection> oc_alctpretrigger(new CSCALCTPreTriggerDigiCollection);
  std::unique_ptr<CSCCLCTPreTriggerCollection> oc_pretrig(new CSCCLCTPreTriggerCollection);
  std::unique_ptr<CSCCorrelatedLCTDigiCollection> oc_lct(new CSCCorrelatedLCTDigiCollection);
  std::unique_ptr<CSCCorrelatedLCTDigiCollection> oc_sorted_lct(new CSCCorrelatedLCTDigiCollection);
  std::unique_ptr<CSCShowerDigiCollection> oc_shower(new CSCShowerDigiCollection);
  std::unique_ptr<CSCShowerDigiCollection> oc_shower_anode(new CSCShowerDigiCollection);
  std::unique_ptr<GEMCoPadDigiCollection> oc_gemcopad(new GEMCoPadDigiCollection);

  if (!wireDigis.isValid()) {
    edm::LogWarning("CSCTriggerPrimitivesProducer|NoInputCollection")
        << "+++ Warning: Collection of wire digis with label " << wireDigiProducer_.label()
        << " requested in configuration, but not found in the event..."
        << " Skipping production of CSC TP digis +++\n";
  }
  if (!compDigis.isValid()) {
    edm::LogWarning("CSCTriggerPrimitivesProducer|NoInputCollection")
        << "+++ Warning: Collection of comparator digis with label " << compDigiProducer_.label()
        << " requested in configuration, but not found in the event..."
        << " Skipping production of CSC TP digis +++\n";
  }
  // Fill output collections if valid input collections are available.
  if (wireDigis.isValid() && compDigis.isValid()) {
    const CSCBadChambers* temp = checkBadChambers_ ? pBadChambers.product() : new CSCBadChambers;
    builder_->build(temp,
                    wireDigis.product(),
                    compDigis.product(),
                    gemPadClusters,
                    *oc_alct,
                    *oc_clct,
                    *oc_alctpretrigger,
                    *oc_clctpretrigger,
                    *oc_pretrig,
                    *oc_lct,
                    *oc_sorted_lct,
                    *oc_shower,
                    *oc_shower_anode,
                    *oc_gemcopad);
    if (!checkBadChambers_)
      delete temp;
  }

  // Put collections in event.
  ev.put(std::move(oc_alct));
  ev.put(std::move(oc_clct));
  if (keepALCTPreTriggers_) {
    ev.put(std::move(oc_alctpretrigger));
  }
  if (keepCLCTPreTriggers_) {
    ev.put(std::move(oc_clctpretrigger));
  }
  ev.put(std::move(oc_pretrig));
  ev.put(std::move(oc_lct));
  ev.put(std::move(oc_sorted_lct), "MPCSORTED");
  if (keepShowers_) {
    ev.put(std::move(oc_shower));
    ev.put(std::move(oc_shower_anode), "Anode");
  }
  // only put GEM copad collections in the event when the
  // integrated local triggers are running
  if (runME11ILT_ or runME21ILT_)
    ev.put(std::move(oc_gemcopad));
}
