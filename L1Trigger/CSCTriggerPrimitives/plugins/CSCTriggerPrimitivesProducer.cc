//-------------------------------------------------
//
//   Class: CSCTriggerPrimitivesProducer
//
//   Description: Steering routine of the local Level-1 Cathode Strip Chamber
//                trigger.
//
//   Author List: S. Valuev, UCLA.
//
//   Modifications:
//
//--------------------------------------------------
 
#include "L1Trigger/CSCTriggerPrimitives/plugins/CSCTriggerPrimitivesProducer.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesBuilder.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCSCCoPadDigiCollection.h"

// Configuration via EventSetup
#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"
#include "CondFormats/DataRecord/interface/CSCDBL1TPParametersRcd.h"


CSCTriggerPrimitivesProducer::CSCTriggerPrimitivesProducer(const edm::ParameterSet& conf) : iev(0) {

  // if false, parameters will be read in from DB using EventSetup mechanism
  // else will use all parameters from the config file
  debugParameters_ = conf.getParameter<bool>("debugParameters");

  wireDigiProducer_ = conf.getParameter<edm::InputTag>("CSCWireDigiProducer");
  compDigiProducer_ = conf.getParameter<edm::InputTag>("CSCComparatorDigiProducer");
  gemPadProducer_ = conf.existsAs<edm::InputTag>("GEMCSCPadDigiProducer")?conf.getParameter<edm::InputTag>("GEMCSCPadDigiProducer"):edm::InputTag("");
  rpcDigiProducer_ = conf.existsAs<edm::InputTag>("RPCDigiProducer")?conf.getParameter<edm::InputTag>("RPCDigiProducer"):edm::InputTag("");
  checkBadChambers_ = conf.getParameter<bool>("checkBadChambers");

  lctBuilder_ = new CSCTriggerPrimitivesBuilder(conf); // pass on the conf

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()){
    throw cms::Exception("Configuration")
      << "GEMDigiProducer::GEMDigiProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
      << "Add the service in the configuration file or remove the modules that require it.";
  }
  CLHEP::HepRandomEngine& engine = rng->getEngine();
  lctBuilder_->setRandomEngine(engine);

  // register what this produces
  produces<CSCALCTDigiCollection>();
  produces<CSCCLCTDigiCollection>();
  produces<CSCCLCTPreTriggerCollection>();
  produces<CSCCorrelatedLCTDigiCollection>();
  produces<CSCCorrelatedLCTDigiCollection>("MPCSORTED");
  produces<GEMCSCCoPadDigiCollection>();
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
    lctBuilder_->setCSCGeometry(&*h);

    edm::ESHandle<GEMGeometry> h_gem;
    try {
      setup.get<MuonGeometryRecord>().get(h_gem);
      lctBuilder_->setGEMGeometry(&*h_gem);
    } catch (edm::eventsetup::NoProxyException<GEMGeometry>& e) {
      edm::LogInfo("L1CSCTPEmulatorNoGEMGeometry") 
	<< "+++ Info: GEM geometry is unavailable. Running CSC-only trigger algorithm. +++\n";
    }

    edm::ESHandle<RPCGeometry> h_rpc;
    try {
      setup.get<MuonGeometryRecord>().get(h_rpc);
      lctBuilder_->setRPCGeometry(&*h_rpc);
    } catch (edm::eventsetup::NoProxyException<RPCGeometry>& e) {
      edm::LogInfo("L1CSCTPEmulatorNoRPCGeometry") 
	<< "+++ Info: RPC geometry is unavailable. Running CSC-only trigger algorithm. +++\n";
    }

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

  const GEMCSCPadDigiCollection *gemPads = nullptr;
  if (!gemPadProducer_.label().empty()) {
    edm::Handle<GEMCSCPadDigiCollection> h_pads;
    ev.getByLabel(gemPadProducer_, h_pads);
    gemPads = h_pads.product();
  }

  const RPCDigiCollection *rpcDigis = nullptr;
  if (!rpcDigiProducer_.label().empty()) {
    edm::Handle<RPCDigiCollection> h_rpcs;
    ev.getByLabel(rpcDigiProducer_, h_rpcs);
    rpcDigis = h_rpcs.product();
  }

  // Create empty collections of ALCTs, CLCTs, and correlated LCTs upstream
  // and downstream of MPC.
  std::auto_ptr<CSCALCTDigiCollection> oc_alct(new CSCALCTDigiCollection);
  std::auto_ptr<CSCCLCTDigiCollection> oc_clct(new CSCCLCTDigiCollection);
  std::auto_ptr<CSCCLCTPreTriggerCollection> oc_pretrig(new CSCCLCTPreTriggerCollection);
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> oc_lct(new CSCCorrelatedLCTDigiCollection);
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> oc_sorted_lct(new CSCCorrelatedLCTDigiCollection);
  std::auto_ptr<GEMCSCCoPadDigiCollection> oc_gemcopad(new GEMCSCCoPadDigiCollection);

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
    std::shared_ptr<const CSCBadChambers> temp( checkBadChambers_ ?
                                                std::shared_ptr<const CSCBadChambers>{pBadChambers.product(), [](const void*){}} :
                                                std::make_shared<const CSCBadChambers>());
    lctBuilder_->build(temp.get(),
                       wireDigis.product(), compDigis.product(), gemPads, rpcDigis,
                       *oc_alct, *oc_clct, *oc_pretrig, *oc_lct, *oc_sorted_lct, *oc_gemcopad);
  }

  // Put collections in event.
  ev.put(oc_alct);
  ev.put(oc_clct);
  ev.put(oc_pretrig);
  ev.put(oc_lct);
  ev.put(oc_sorted_lct,"MPCSORTED");
  ev.put(oc_gemcopad);
}
