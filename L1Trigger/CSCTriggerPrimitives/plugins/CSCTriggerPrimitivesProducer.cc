/** \class CSCTriggerPrimitivesProducer
 *
 * Implementation of the local Level-1 Cathode Strip Chamber trigger.
 * Simulates functionalities of the anode and cathode Local Charged Tracks
 * (LCT) processors, of the Trigger Mother Board (TMB), and of the Muon Port
 * Card (MPC).
 *
 * Input to the simulation are collections of the CSC wire and comparator
 * digis.
 *
 * Produces four collections of the Level-1 CSC Trigger Primitives (track
 * stubs, or LCTs): anode LCTs (ALCTs), cathode LCTs (CLCTs), correlated
 * LCTs at TMB, and correlated LCTs at MPC.
 *
 * \author Slava Valuev, UCLA.
 *
 * The trigger primitive emulator has been expanded with options to
 * use both ALCTs, CLCTs and GEM clusters. The GEM-CSC integrated
 * local trigger combines ALCT, CLCT and GEM information to produce integrated
 * stubs. The available stub types can be found in the class definition of
 * CSCCorrelatedLCTDigi (DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h)
 *
 * authors: Sven Dildick (TAMU), Tao Huang (TAMU)
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTPreTriggerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCTriggerPrimitivesBuilder.h"
#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"
#include "CondFormats/DataRecord/interface/CSCL1TPLookupTableCCLUTRcd.h"
#include "CondFormats/DataRecord/interface/CSCL1TPLookupTableME11ILTRcd.h"
#include "CondFormats/DataRecord/interface/CSCL1TPLookupTableME21ILTRcd.h"
#include "CondFormats/DataRecord/interface/CSCDBL1TPParametersRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

// temporarily switch to a "one" module with a CSCTriggerPrimitivesBuilder data member
class CSCTriggerPrimitivesProducer : public edm::stream::EDProducer<> {
public:
  explicit CSCTriggerPrimitivesProducer(const edm::ParameterSet&);
  ~CSCTriggerPrimitivesProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // temporarily switch to a "one" module with a CSCTriggerPrimitivesBuilder data member
  std::unique_ptr<CSCTriggerPrimitivesBuilder> builder_;

  // input tags for input collections
  edm::InputTag compDigiProducer_;
  edm::InputTag wireDigiProducer_;
  edm::InputTag gemPadDigiClusterProducer_;

  // tokens
  edm::EDGetTokenT<CSCComparatorDigiCollection> comp_token_;
  edm::EDGetTokenT<CSCWireDigiCollection> wire_token_;
  edm::EDGetTokenT<GEMPadDigiClusterCollection> gem_pad_cluster_token_;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemToken_;
  edm::ESGetToken<CSCBadChambers, CSCBadChambersRcd> pBadChambersToken_;
  edm::ESGetToken<CSCL1TPLookupTableCCLUT, CSCL1TPLookupTableCCLUTRcd> pLookupTableCCLUTToken_;
  edm::ESGetToken<CSCL1TPLookupTableME11ILT, CSCL1TPLookupTableME11ILTRcd> pLookupTableME11ILTToken_;
  edm::ESGetToken<CSCL1TPLookupTableME21ILT, CSCL1TPLookupTableME21ILTRcd> pLookupTableME21ILTToken_;
  edm::ESGetToken<CSCDBL1TPParameters, CSCDBL1TPParametersRcd> confToken_;

  std::unique_ptr<CSCBadChambers const> dummyBadChambers_;
  // switch to force the use of parameters from config file rather then from DB
  bool debugParameters_;

  // switch to for enabling checking against the list of bad chambers
  bool checkBadChambers_;

  // Write out pre-triggers
  bool keepCLCTPreTriggers_;
  bool keepALCTPreTriggers_;

  // write out showrs
  bool keepShowers_;

  // switches to enable the Run-3 pattern finding
  bool runCCLUT_;       // 'OR' of the two options below
  bool runCCLUT_TMB_;   // ME1/2, ME1/3, ME2/2, ME3/2, ME4/2
  bool runCCLUT_OTMB_;  // ME1/1, ME2/1, ME3/1, ME4/1

  bool runILT_;  // // 'OR' of the two options below
  bool runME11ILT_;
  bool runME21ILT_;
};

CSCTriggerPrimitivesProducer::CSCTriggerPrimitivesProducer(const edm::ParameterSet& conf) {
  // if false, parameters will be read in from DB using EventSetup mechanism
  // else will use all parameters from the config file
  debugParameters_ = conf.getParameter<bool>("debugParameters");

  wireDigiProducer_ = conf.getParameter<edm::InputTag>("CSCWireDigiProducer");
  compDigiProducer_ = conf.getParameter<edm::InputTag>("CSCComparatorDigiProducer");
  gemPadDigiClusterProducer_ = conf.getParameter<edm::InputTag>("GEMPadDigiClusterProducer");

  checkBadChambers_ = conf.getParameter<bool>("checkBadChambers");
  if (not checkBadChambers_) {
    dummyBadChambers_ = std::make_unique<CSCBadChambers>();
  } else {
    pBadChambersToken_ = esConsumes<CSCBadChambers, CSCBadChambersRcd>();
  }

  keepCLCTPreTriggers_ = conf.getParameter<bool>("keepCLCTPreTriggers");
  keepALCTPreTriggers_ = conf.getParameter<bool>("keepALCTPreTriggers");
  keepShowers_ = conf.getParameter<bool>("keepShowers");

  // check whether you need to run the integrated local triggers
  const edm::ParameterSet& commonParam = conf.getParameter<edm::ParameterSet>("commonParam");
  runCCLUT_TMB_ = commonParam.getParameter<bool>("runCCLUT_TMB");
  runCCLUT_OTMB_ = commonParam.getParameter<bool>("runCCLUT_OTMB");
  runCCLUT_ = runCCLUT_TMB_ or runCCLUT_OTMB_;

  runME11ILT_ = commonParam.getParameter<bool>("runME11ILT");
  runME21ILT_ = commonParam.getParameter<bool>("runME21ILT");
  runILT_ = runME11ILT_ or runME21ILT_;

  wire_token_ = consumes<CSCWireDigiCollection>(wireDigiProducer_);
  comp_token_ = consumes<CSCComparatorDigiCollection>(compDigiProducer_);
  if (runILT_) {
    gem_pad_cluster_token_ = consumes<GEMPadDigiClusterCollection>(gemPadDigiClusterProducer_);
    gemToken_ = esConsumes<GEMGeometry, MuonGeometryRecord>();
  }

  cscToken_ = esConsumes<CSCGeometry, MuonGeometryRecord>();
  // consume lookup tables only when flags are set
  if (runCCLUT_)
    pLookupTableCCLUTToken_ = esConsumes<CSCL1TPLookupTableCCLUT, CSCL1TPLookupTableCCLUTRcd>();
  if (runME11ILT_)
    pLookupTableME11ILTToken_ = esConsumes<CSCL1TPLookupTableME11ILT, CSCL1TPLookupTableME11ILTRcd>();
  if (runME21ILT_)
    pLookupTableME21ILTToken_ = esConsumes<CSCL1TPLookupTableME21ILT, CSCL1TPLookupTableME21ILTRcd>();
  if (not debugParameters_)
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
    produces<CSCShowerDigiCollection>("Cathode");
  }
  if (runILT_) {
    produces<GEMCoPadDigiCollection>();
  }

  builder_ = std::make_unique<CSCTriggerPrimitivesBuilder>(conf);
}

CSCTriggerPrimitivesProducer::~CSCTriggerPrimitivesProducer() {}

void CSCTriggerPrimitivesProducer::produce(edm::Event& ev, const edm::EventSetup& setup) {
  auto* builder = builder_.get();

  // get the csc geometry
  builder->setCSCGeometry(&setup.getData(cscToken_));

  // get the gem geometry if it's there
  if (runILT_) {
    edm::ESHandle<GEMGeometry> h_gem = setup.getHandle(gemToken_);
    if (h_gem.isValid()) {
      builder->setGEMGeometry(&*h_gem);
    } else {
      edm::LogWarning("CSCTriggerPrimitivesProducer|NoGEMGeometry")
          << "GEM geometry is unavailable. Running CSC-only trigger algorithm. +++\n";
    }
  }

  if (runCCLUT_) {
    edm::ESHandle<CSCL1TPLookupTableCCLUT> conf = setup.getHandle(pLookupTableCCLUTToken_);
    if (conf.product() == nullptr) {
      edm::LogError("CSCTriggerPrimitivesProducer")
          << "Failed to find a CSCL1TPLookupTableCCLUTRcd in EventSetup with runCCLUT_ on";
      return;
    }
    builder->setESLookupTables(conf.product());
  }

  if (runME11ILT_) {
    edm::ESHandle<CSCL1TPLookupTableME11ILT> conf = setup.getHandle(pLookupTableME11ILTToken_);
    if (conf.product() == nullptr) {
      edm::LogError("CSCTriggerPrimitivesProducer")
          << "Failed to find a CSCL1TPLookupTableME11ILTRcd in EventSetup with runME11ILT_ on";
      return;
    }
    builder->setESLookupTables(conf.product());
  }

  if (runME21ILT_) {
    edm::ESHandle<CSCL1TPLookupTableME21ILT> conf = setup.getHandle(pLookupTableME21ILTToken_);
    if (conf.product() == nullptr) {
      edm::LogError("CSCTriggerPrimitivesProducer")
          << "Failed to find a CSCL1TPLookupTableME21ILTRcd in EventSetup with runME21ILT_ on";
      return;
    }
    builder->setESLookupTables(conf.product());
  }

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
    builder->setConfigParameters(conf.product());
  }

  // Get the collections of comparator & wire digis from event.
  edm::Handle<CSCComparatorDigiCollection> compDigis;
  edm::Handle<CSCWireDigiCollection> wireDigis;
  ev.getByToken(comp_token_, compDigis);
  ev.getByToken(wire_token_, wireDigis);

  // input GEM pad cluster collection for upgrade scenarios
  edm::Handle<GEMPadDigiClusterCollection> gemPadDigiClusters;
  const GEMPadDigiClusterCollection* gemPadClusters = nullptr;

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
  std::unique_ptr<CSCShowerDigiCollection> oc_shower_cathode(new CSCShowerDigiCollection);
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
  // the GEM-CSC trigger flag is set, so GEM clusters are expected in the event data
  if (runILT_) {
    // no valid label, let the user know that GEM clusters are missing
    // the algorithm should not crash. instead it should just produce the regular CSC LCTs
    // in ME1/1 and/or ME2/1
    ev.getByToken(gem_pad_cluster_token_, gemPadDigiClusters);
    if (!gemPadDigiClusters.isValid()) {
      edm::LogWarning("CSCTriggerPrimitivesProducer|NoInputCollection")
          << "+++ Warning: Collection of GEM clusters with label " << gemPadDigiClusterProducer_.label()
          << " requested in configuration, but not found in the event..."
          << " Running CSC-only trigger algorithm +++\n";
    } else {
      // when the GEM-CSC trigger should be run and the label is not empty, set a valid pointer
      gemPadClusters = gemPadDigiClusters.product();
    }
  }

  // Fill output collections if valid input collections are available.
  if (wireDigis.isValid() && compDigis.isValid()) {
    const CSCBadChambers* temp = nullptr;
    if (checkBadChambers_) {
      // Find conditions data for bad chambers.
      temp = &setup.getData(pBadChambersToken_);
    } else {
      temp = dummyBadChambers_.get();
    }
    builder->build(temp,
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
                   *oc_shower_anode,
                   *oc_shower_cathode,
                   *oc_shower,
                   *oc_gemcopad);
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
    ev.put(std::move(oc_shower_cathode), "Cathode");
  }
  // only put GEM copad collections in the event when the
  // integrated local triggers are running
  if (runILT_)
    ev.put(std::move(oc_gemcopad));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CSCTriggerPrimitivesProducer);
