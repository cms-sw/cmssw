#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDigiToRawAccept.h"

#include <memory>
#include <iostream>

class CSCPackerUnpackerUnitTest : public edm::one::EDAnalyzer<> {
public:
  explicit CSCPackerUnpackerUnitTest(const edm::ParameterSet&);
  ~CSCPackerUnpackerUnitTest() override {}

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyzeChamber(const CSCDetId& cscId,
                      const CSCWireDigiCollection& wd,
                      const CSCWireDigiCollection& wdu,
                      const CSCStripDigiCollection& sd,
                      const CSCStripDigiCollection& sdu,
                      const CSCComparatorDigiCollection& cd,
                      const CSCComparatorDigiCollection& cdu,
                      const CSCALCTDigiCollection& alct,
                      const CSCCLCTDigiCollection& clct,
                      const CSCCLCTPreTriggerDigiCollection& pre,
                      const CSCCorrelatedLCTDigiCollection& lct,
                      const CSCShowerDigiCollection* showers) const;

  // the return value indicates the number of failed tests
  unsigned analyzeALCT(const CSCDetId& cscDetId,
                       const CSCWireDigiCollection& wires,
                       const CSCWireDigiCollection& wires_unpacked,
                       const CSCALCTDigiCollection& alcts) const;
  unsigned analyzeCLCT(const CSCDetId& cscDetId,
                       const CSCComparatorDigiCollection& comparators,
                       const CSCComparatorDigiCollection& comparators_unpacked,
                       const CSCCLCTDigiCollection& clcts) const;
  unsigned analyzePreCLCT(const CSCDetId& cscDetId,
                          const CSCStripDigiCollection& strips,
                          const CSCStripDigiCollection& strips_unpacked,
                          const CSCCLCTPreTriggerDigiCollection& preclcts) const;

  // helper functions
  template <typename DIGICollection>
  unsigned nDigis(const CSCDetId& cscId, const DIGICollection& digis) const;

  template <typename DIGICollection>
  unsigned nDigisLayer(const CSCDetId& cscId, const DIGICollection& digis) const;

  unsigned nDigisCFEB(const CSCDetId& cscId, const CSCStripDigiCollection& digis, int cfeb) const;

  unsigned nDigisCFEBLayer(const CSCDetId& cscId, const CSCStripDigiCollection& digis, int cfeb) const;

  unsigned getNCFEBs(unsigned type) const;

  int alctWindowMin_;
  int alctWindowMax_;
  int clctWindowMin_;
  int clctWindowMax_;
  int preTriggerWindowMin_;
  int preTriggerWindowMax_;
  unsigned formatVersion_;
  bool packEverything_;
  bool usePreTriggers_;
  bool useCSCShowers_;
  bool packByCFEB_;
  bool testALCT_;
  bool testCLCT_;
  bool testPreCLCT_;

  edm::EDGetTokenT<CSCWireDigiCollection> wd_token_;
  edm::EDGetTokenT<CSCWireDigiCollection> wd_unpacked_token_;
  edm::EDGetTokenT<CSCStripDigiCollection> sd_token_;
  edm::EDGetTokenT<CSCStripDigiCollection> sd_unpacked_token_;
  edm::EDGetTokenT<CSCComparatorDigiCollection> cd_token_;
  edm::EDGetTokenT<CSCComparatorDigiCollection> cd_unpacked_token_;
  edm::EDGetTokenT<CSCALCTDigiCollection> al_token_;
  edm::EDGetTokenT<CSCCLCTDigiCollection> cl_token_;
  edm::EDGetTokenT<CSCCLCTPreTriggerDigiCollection> clpre_token_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> co_token_;
  edm::EDGetTokenT<CSCShowerDigiCollection> shower_token;
};

CSCPackerUnpackerUnitTest::CSCPackerUnpackerUnitTest(const edm::ParameterSet& conf)
    : alctWindowMin_(conf.getParameter<int>("alctWindowMin")),
      alctWindowMax_(conf.getParameter<int>("alctWindowMax")),
      clctWindowMin_(conf.getParameter<int>("clctWindowMin")),
      clctWindowMax_(conf.getParameter<int>("clctWindowMax")),
      preTriggerWindowMin_(conf.getParameter<int>("preTriggerWindowMin")),
      preTriggerWindowMax_(conf.getParameter<int>("preTriggerWindowMax")),
      formatVersion_(conf.getParameter<unsigned>("formatVersion")),
      packEverything_(conf.getParameter<bool>("packEverything")),
      usePreTriggers_(conf.getParameter<bool>("usePreTriggers")),
      useCSCShowers_(conf.getParameter<bool>("useCSCShowers")),
      packByCFEB_(conf.getParameter<bool>("packByCFEB")),
      testALCT_(conf.getParameter<bool>("testALCT")),
      testCLCT_(conf.getParameter<bool>("testCLCT")),
      testPreCLCT_(conf.getParameter<bool>("testPreCLCT")),
      wd_token_(consumes<CSCWireDigiCollection>(conf.getParameter<edm::InputTag>("wireTag"))),
      wd_unpacked_token_(consumes<CSCWireDigiCollection>(conf.getParameter<edm::InputTag>("wireUnpackedTag"))),
      sd_token_(consumes<CSCStripDigiCollection>(conf.getParameter<edm::InputTag>("stripTag"))),
      sd_unpacked_token_(consumes<CSCStripDigiCollection>(conf.getParameter<edm::InputTag>("stripUnpackedTag"))),
      cd_token_(consumes<CSCComparatorDigiCollection>(conf.getParameter<edm::InputTag>("comparatorTag"))),
      cd_unpacked_token_(
          consumes<CSCComparatorDigiCollection>(conf.getParameter<edm::InputTag>("comparatorUnpackedTag"))),
      al_token_(consumes<CSCALCTDigiCollection>(conf.getParameter<edm::InputTag>("alctTag"))),
      cl_token_(consumes<CSCCLCTDigiCollection>(conf.getParameter<edm::InputTag>("clctTag"))),
      clpre_token_(consumes<CSCCLCTPreTriggerDigiCollection>(conf.getParameter<edm::InputTag>("clctpreTag"))),
      co_token_(consumes<CSCCorrelatedLCTDigiCollection>(conf.getParameter<edm::InputTag>("corrclctTag"))) {
  if (useCSCShowers_) {
    shower_token = consumes<CSCShowerDigiCollection>(conf.getParameter<edm::InputTag>("showerDigiTag"));
  }
}

void CSCPackerUnpackerUnitTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // digi collections
  desc.add<edm::InputTag>("wireTag", edm::InputTag("simMuonCSCDigis", "MuonCSCWireDigi"));
  desc.add<edm::InputTag>("stripTag", edm::InputTag("simMuonCSCDigis", "MuonCSCStripDigi"));
  desc.add<edm::InputTag>("comparatorTag", edm::InputTag("simMuonCSCDigis", "MuonCSCComparatorDigi"));
  // DAQ collections
  desc.add<edm::InputTag>("wireUnpackedTag", edm::InputTag("muonCSCDigis", "MuonCSCWireDigi"));
  desc.add<edm::InputTag>("stripUnpackedTag", edm::InputTag("muonCSCDigis", "MuonCSCStripDigi"));
  desc.add<edm::InputTag>("comparatorUnpackedTag", edm::InputTag("muonCSCDigis", "MuonCSCComparatorDigi"));
  // trigger collections
  desc.add<edm::InputTag>("alctTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("clctTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("clctpreTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("corrclctTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("showerDigiTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  // readout windows
  desc.add<int32_t>("alctWindowMin", -3);
  desc.add<int32_t>("alctWindowMax", 3);
  desc.add<int32_t>("clctWindowMin", -3);
  desc.add<int32_t>("clctWindowMax", 3);
  desc.add<int32_t>("preTriggerWindowMin", -3);
  desc.add<int32_t>("preTriggerWindowMax", 1);
  desc.add<unsigned int>("formatVersion", 2005);
  // enable functions
  desc.add<bool>("testALCT", true);
  desc.add<bool>("testCLCT", true);
  desc.add<bool>("testPreCLCT", true);
  desc.add<bool>("usePreTriggers", true);
  desc.add<bool>("packEverything", false);
  desc.add<bool>("useCSCShowers", false);
  desc.add<bool>("packByCFEB", true);
  descriptions.add("cscPackerUnpackerUnitTestDef", desc);
}

void CSCPackerUnpackerUnitTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // handles, tokens
  edm::Handle<CSCWireDigiCollection> wires;
  iEvent.getByToken(wd_token_, wires);

  edm::Handle<CSCWireDigiCollection> wires_unpacked;
  iEvent.getByToken(wd_unpacked_token_, wires_unpacked);

  edm::Handle<CSCStripDigiCollection> strips;
  iEvent.getByToken(sd_token_, strips);

  edm::Handle<CSCStripDigiCollection> strips_unpacked;
  iEvent.getByToken(sd_unpacked_token_, strips_unpacked);

  edm::Handle<CSCComparatorDigiCollection> comparators;
  iEvent.getByToken(cd_token_, comparators);

  edm::Handle<CSCComparatorDigiCollection> comparators_unpacked;
  iEvent.getByToken(cd_unpacked_token_, comparators_unpacked);

  edm::Handle<CSCALCTDigiCollection> alcts;
  iEvent.getByToken(al_token_, alcts);

  edm::Handle<CSCCLCTDigiCollection> clcts;
  iEvent.getByToken(cl_token_, clcts);

  edm::Handle<CSCCLCTPreTriggerDigiCollection> preclcts;
  iEvent.getByToken(clpre_token_, preclcts);

  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts;
  iEvent.getByToken(co_token_, lcts);

  const CSCShowerDigiCollection* cscShowerDigisPtr = nullptr;
  if (useCSCShowers_) {
    cscShowerDigisPtr = &iEvent.get(shower_token);
  }

  const int min_endcap = CSCDetId::minEndcapId();
  const int max_endcap = CSCDetId::maxEndcapId();
  const int min_station = CSCDetId::minStationId();
  const int max_station = CSCDetId::maxStationId();
  const int min_sector = CSCTriggerNumbering::minTriggerSectorId();
  const int max_sector = CSCTriggerNumbering::maxTriggerSectorId();
  const int min_subsector = CSCTriggerNumbering::minTriggerSubSectorId();
  const int max_subsector = CSCTriggerNumbering::maxTriggerSubSectorId();
  const int min_chamber = CSCTriggerNumbering::minTriggerCscId();
  const int max_chamber = CSCTriggerNumbering::maxTriggerCscId();

  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
        for (int subs = min_subsector; subs <= numsubs; subs++) {
          for (int cham = min_chamber; cham <= max_chamber; cham++) {
            int ring = CSCTriggerNumbering::ringFromTriggerLabels(stat, cham);

            // actual chamber number =/= trigger chamber number
            int chid = CSCTriggerNumbering::chamberFromTriggerLabels(sect, subs, stat, cham);

            // 0th layer means whole chamber.
            CSCDetId cscDetId(endc, stat, ring, chid, 0);

            analyzeChamber(cscDetId,
                           *wires,
                           *wires_unpacked,
                           *strips,
                           *strips_unpacked,
                           *comparators,
                           *comparators_unpacked,
                           *alcts,
                           *clcts,
                           *preclcts,
                           *lcts,
                           cscShowerDigisPtr);
          }
        }
      }
    }
  }
}

void CSCPackerUnpackerUnitTest::analyzeChamber(const CSCDetId& cscDetId,
                                               const CSCWireDigiCollection& wires,
                                               const CSCWireDigiCollection& wires_unpacked,
                                               const CSCStripDigiCollection& strips,
                                               const CSCStripDigiCollection& strips_unpacked,
                                               const CSCComparatorDigiCollection& comparators,
                                               const CSCComparatorDigiCollection& comparators_unpacked,
                                               const CSCALCTDigiCollection& alcts,
                                               const CSCCLCTDigiCollection& clcts,
                                               const CSCCLCTPreTriggerDigiCollection& preclcts,
                                               const CSCCorrelatedLCTDigiCollection& lcts,
                                               const CSCShowerDigiCollection* showers) const {
  if (testALCT_) {
    const unsigned nFailedTestsALCT = analyzeALCT(cscDetId, wires, wires_unpacked, alcts);
    if (nFailedTestsALCT) {
      edm::LogWarning("CSCPackerUnpackerUnitTest")
          << nFailedTestsALCT << " ALCT test(s) failed in " << cscDetId.chamberName();
    }
  }

  if (testCLCT_) {
    const unsigned nFailedTestsCLCT = analyzeCLCT(cscDetId, comparators, comparators_unpacked, clcts);
    if (nFailedTestsCLCT) {
      edm::LogWarning("CSCPackerUnpackerUnitTest")
          << nFailedTestsCLCT << " CLCT test(s) failed in " << cscDetId.chamberName();
    }
  }

  if (testPreCLCT_) {
    const unsigned nFailedTestsPreCLCT = analyzePreCLCT(cscDetId, strips, strips_unpacked, preclcts);
    if (nFailedTestsPreCLCT) {
      edm::LogWarning("CSCPackerUnpackerUnitTest")
          << nFailedTestsPreCLCT << " PreCLCT test(s) failed in " << cscDetId.chamberName();
    }
  }
}

unsigned CSCPackerUnpackerUnitTest::analyzeALCT(const CSCDetId& cscDetId,
                                                const CSCWireDigiCollection& wires,
                                                const CSCWireDigiCollection& wires_unpacked,
                                                const CSCALCTDigiCollection& alcts) const {
  unsigned numWireDigis = nDigis(cscDetId, wires);
  // no simulated wire digis means that all tests pass
  if (numWireDigis == 0) {
    return 0;
  }

  // readout condition for wires: L1A + ALCT
  bool hasALCT =
      CSCDigiToRawAccept::accept(cscDetId, alcts, alctWindowMin_, alctWindowMax_, CSCConstants::ALCT_CENTRAL_BX);

  unsigned numWireDigisUnpacked = nDigis(cscDetId, wires_unpacked);

  unsigned testsFailed = 0;

  // these tests are only done when the trigger/pretriggers are enabled
  if (!packEverything_ and usePreTriggers_) {
    // test 1: check that an ALCT in this chamber kept the wire digis
    if (hasALCT) {
      if (numWireDigisUnpacked == 0) {
        testsFailed++;
        edm::LogWarning("analyzeALCT::Test1Failure") << "hasALCT " << hasALCT << " numWireDigisUnpacked "
                                                     << numWireDigisUnpacked << " " << cscDetId.chamberName();
      }
    }

    // test 2: check no wire digis were kept if there was no ALCT
    if (numWireDigisUnpacked != 0) {
      if (!hasALCT) {
        testsFailed++;
        edm::LogWarning("analyzeALCT::Test2Failure") << "hasALCT " << hasALCT << " numWireDigisUnpacked "
                                                     << numWireDigisUnpacked << " " << cscDetId.chamberName();
      }
    }
  }

  // test 3: count the number of wire digis before and after unpacking
  if (numWireDigis < numWireDigisUnpacked) {
    testsFailed++;
    edm::LogWarning("analyzeALCT::Test3Failure") << "numWireDigis " << numWireDigis << " numWireDigisUnpacked "
                                                 << numWireDigisUnpacked << " " << cscDetId.chamberName();
  }

  return testsFailed;
}

unsigned CSCPackerUnpackerUnitTest::analyzeCLCT(const CSCDetId& cscDetId,
                                                const CSCComparatorDigiCollection& comparators,
                                                const CSCComparatorDigiCollection& comparators_unpacked,
                                                const CSCCLCTDigiCollection& clcts) const {
  unsigned numCompDigis = nDigis(cscDetId, comparators);
  // no simulated comparator digis means that all tests pass
  if (numCompDigis == 0) {
    return 0;
  }

  // readout condition for comparators in (O)TMB; L1A + CLCT
  bool hasCLCT =
      CSCDigiToRawAccept::accept(cscDetId, clcts, clctWindowMin_, clctWindowMax_, CSCConstants::CLCT_CENTRAL_BX);

  unsigned numCompDigisUnpacked = nDigis(cscDetId, comparators_unpacked);

  unsigned testsFailed = 0;

  // these tests are only done when the trigger/pretriggers are enabled
  if (!packEverything_ and usePreTriggers_) {
    // test 1: check that an CLCT in this chamber kept the comp digis
    if (hasCLCT) {
      if (numCompDigisUnpacked == 0) {
        testsFailed++;
        edm::LogWarning("analyzeCLCT::Test1Failure") << "hasCLCT " << hasCLCT << " numCompDigisUnpacked "
                                                     << numCompDigisUnpacked << " " << cscDetId.chamberName();
      }
    }

    // test 2: check no comp digis were kept if there was no CLCT
    if (numCompDigisUnpacked != 0) {
      if (!hasCLCT) {
        testsFailed++;
        edm::LogWarning("analyzeCLCT::Test2Failure") << "hasCLCT " << hasCLCT << " numCompDigisUnpacked "
                                                     << numCompDigisUnpacked << " " << cscDetId.chamberName();
      }
    }
  }

  // test 3: count the number of comp digis before and after unpacking
  if (numCompDigis < numCompDigisUnpacked) {
    testsFailed++;
    edm::LogWarning("analyzeCLCT::Test3Failure") << "numCompDigis " << numCompDigis << " numCompDigisUnpacked "
                                                 << numCompDigisUnpacked << " " << cscDetId.chamberName();
  }

  return testsFailed;
}

unsigned CSCPackerUnpackerUnitTest::analyzePreCLCT(const CSCDetId& cscDetId,
                                                   const CSCStripDigiCollection& strips,
                                                   const CSCStripDigiCollection& strips_unpacked,
                                                   const CSCCLCTPreTriggerDigiCollection& preclcts) const {
  unsigned numStripDigis = nDigis(cscDetId, strips);
  // no simulated strip digis means that all tests pass
  if (numStripDigis == 0) {
    return 0;
  }

  // determine where the pretriggers are
  std::vector<bool> preTriggerInCFEB;
  preTriggerInCFEB.resize(CSCConstants::MAX_CFEBS_RUN2);

  // readout condition for strips: L1A + preCLCT in CFEB
  bool hasPreCLCT = CSCDigiToRawAccept::accept(
      cscDetId, preclcts, preTriggerWindowMin_, preTriggerWindowMax_, CSCConstants::CLCT_CENTRAL_BX, preTriggerInCFEB);

  unsigned testsFailed = 0;

  unsigned nStripDigisUnpacked = nDigis(cscDetId, strips_unpacked);

  // these tests are only done when the trigger/pretriggers are enabled
  if (!packEverything_ and usePreTriggers_) {
    // Test when NOT packing by CFEB
    if (!packByCFEB_) {
      // test 1: check that a PreCLCT in this chamber kept the strip digis
      if (hasPreCLCT) {
        if (nStripDigisUnpacked == 0) {
          testsFailed++;
          edm::LogWarning("analyzePreCLCT::Test1Failure") << "preTrigger " << hasPreCLCT << " nStripDigisUnpacked "
                                                          << nStripDigisUnpacked << " " << cscDetId.chamberName();
        }
      }

      // test 2: check no strip digis were kept if there was no PreCLCT
      if (nStripDigisUnpacked != 0) {
        if (!hasPreCLCT) {
          testsFailed++;
          edm::LogWarning("analyzePreCLCT::Test2Failure") << "preTrigger " << hasPreCLCT << " nStripDigisUnpacked "
                                                          << nStripDigisUnpacked << " " << cscDetId.chamberName();
        }
      }
    }

    // Test when packing by CFEBs
    else {
      // tests are performed for each CFEB
      for (unsigned i = 0; i < CSCConstants::MAX_CFEBS_RUN2; i++) {
        // only do the tests for the CFEBs in this chamber (4,5,7)
        if (i >= getNCFEBs(cscDetId.iChamberType() - 1))
          continue;

        unsigned nStripDigisUnpackedCFEB = nDigisCFEB(cscDetId, strips_unpacked, i);

        // test 1: check that a PreCLCT in this chamber kept the strip digis
        if (preTriggerInCFEB[i]) {
          if (nStripDigisUnpackedCFEB == 0) {
            testsFailed++;
            edm::LogWarning("analyzePreCLCT::Test1Failure")
                << "CFEB " << i << "preTriggerInCFEB " << preTriggerInCFEB[i] << " nStripDigisUnpackedCFEB "
                << nStripDigisUnpackedCFEB << " " << cscDetId.chamberName();
          }
        }

        // test 2: check no strip digis were kept if there was no PreCLCT
        if (nStripDigisUnpackedCFEB != 0) {
          if (!preTriggerInCFEB[i]) {
            testsFailed++;
            edm::LogWarning("analyzePreCLCT::Test2Failure")
                << "CFEB " << i << "preTriggerInCFEB " << preTriggerInCFEB[i] << " nStripDigisUnpackedCFEB "
                << nStripDigisUnpackedCFEB << " " << cscDetId.chamberName();
          }
        }
      }
    }
  }

  return testsFailed;
}

template <typename DIGICollection>
unsigned CSCPackerUnpackerUnitTest::nDigis(const CSCDetId& detid, const DIGICollection& digis) const {
  unsigned nDigis = 0;
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    CSCDetId ldetid(detid.endcap(), detid.station(), detid.ring(), detid.chamber(), i_layer + 1);
    nDigis += nDigisLayer(ldetid, digis);

    if (detid.station() == 1 and detid.ring() == 1) {
      CSCDetId ldetid_me1a(detid.endcap(), detid.station(), 4, detid.chamber(), i_layer + 1);
      nDigis += nDigisLayer(ldetid_me1a, digis);
    }
  }
  return nDigis;
}

template <typename DIGICollection>
unsigned CSCPackerUnpackerUnitTest::nDigisLayer(const CSCDetId& cscId, const DIGICollection& digis) const {
  unsigned nDigis = 0;
  auto digiRange = digis.get(cscId);
  for (auto digiItr = digiRange.first; digiItr != digiRange.second; ++digiItr) {
    nDigis++;
  }
  return nDigis;
}

unsigned CSCPackerUnpackerUnitTest::nDigisCFEB(const CSCDetId& detid,
                                               const CSCStripDigiCollection& digis,
                                               int cfeb) const {
  unsigned nDigis = 0;

  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    // ME1/A case
    if (detid.station() == 1 and detid.ring() == 1 and cfeb >= 4) {
      CSCDetId ldetid(detid.endcap(), detid.station(), 4, detid.chamber(), i_layer + 1);
      nDigis += nDigisCFEBLayer(ldetid, digis, cfeb);
    }
    // all other cases
    else {
      CSCDetId ldetid(detid.endcap(), detid.station(), detid.ring(), detid.chamber(), i_layer + 1);
      nDigis += nDigisCFEBLayer(ldetid, digis, cfeb);
    }
  }
  return nDigis;
}

unsigned CSCPackerUnpackerUnitTest::nDigisCFEBLayer(const CSCDetId& cscId,
                                                    const CSCStripDigiCollection& digis,
                                                    int cfeb) const {
  unsigned nDigis = 0;
  auto digiRange = digis.get(cscId);
  for (auto digiItr = digiRange.first; digiItr != digiRange.second; ++digiItr) {
    CSCStripDigi digi = *digiItr;
    int cfb = digi.getCFEB();
    if (cscId.station() == 1 and cscId.ring() == 4) {
      cfb += CSCConstants::NUM_CFEBS_ME1B;
    }
    if (cfeb == cfb) {
      nDigis++;
    }
  }
  return nDigis;
}

unsigned CSCPackerUnpackerUnitTest::getNCFEBs(unsigned type) const {
  const unsigned cfebs[10] = {CSCConstants::NUM_CFEBS_ME1A_UNGANGED,
                              CSCConstants::NUM_CFEBS_ME11_UNGANGED,
                              CSCConstants::NUM_CFEBS_ME12,
                              CSCConstants::NUM_CFEBS_ME13,
                              CSCConstants::NUM_CFEBS_ME21,
                              CSCConstants::NUM_CFEBS_ME22,
                              CSCConstants::NUM_CFEBS_ME31,
                              CSCConstants::NUM_CFEBS_ME32,
                              CSCConstants::NUM_CFEBS_ME41,
                              CSCConstants::NUM_CFEBS_ME42};
  return cfebs[type];
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCPackerUnpackerUnitTest);
