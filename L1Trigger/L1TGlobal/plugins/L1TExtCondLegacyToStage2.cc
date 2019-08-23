///
/// \class l1t::L1TExtCondLegacyToStage2
///
/// Description: Fill uGT external condition (stage2) with legacy information from data
///
///
/// \author: D. Puigh OSU
/// \revised: V. Rekovic

// system include files

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "L1Trigger/L1TGlobal/plugins/TriggerMenuParser.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

//#include <vector>

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

using namespace std;
using namespace edm;
using namespace l1t;

//
// class declaration
//

class L1TExtCondLegacyToStage2 : public stream::EDProducer<> {
public:
  explicit L1TExtCondLegacyToStage2(const ParameterSet&);
  ~L1TExtCondLegacyToStage2() override;

  static void fillDescriptions(ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  //unsigned long long m_paramsCacheId; // Cache-ID from current parameters, to check if needs to be updated.
  //std::shared_ptr<const CaloParams> m_dbpars; // Database parameters for the trigger, to be updated as needed.
  //std::shared_ptr<const FirmwareVersion> m_fwv;
  //std::shared_ptr<FirmwareVersion> m_fwv; //not const during testing.

  // BX parameters
  int bxFirst_;
  int bxLast_;

  // Readout Record token
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtReadoutRecordToken;

  unsigned long long m_l1GtMenuCacheID;
  std::map<std::string, unsigned int> m_extBitMap;
};

//
// constructors and destructor
//
L1TExtCondLegacyToStage2::L1TExtCondLegacyToStage2(const ParameterSet& iConfig)
    : bxFirst_(iConfig.getParameter<int>("bxFirst")),
      bxLast_(iConfig.getParameter<int>("bxLast")),
      gtReadoutRecordToken(
          consumes<L1GlobalTriggerReadoutRecord>(iConfig.getParameter<edm::InputTag>("LegacyGtReadoutRecord"))) {
  // register what you produce
  produces<GlobalExtBlkBxCollection>();

  m_l1GtMenuCacheID = 0ULL;
}

L1TExtCondLegacyToStage2::~L1TExtCondLegacyToStage2() {}

//
// member functions
//

// ------------ method called to produce the data ------------
void L1TExtCondLegacyToStage2::produce(Event& iEvent, const EventSetup& iSetup) {
  LogDebug("L1TExtCondLegacyToStage2") << "L1TExtCondLegacyToStage2::produce function called...\n";

  // get / update the trigger menu from the EventSetup
  // local cache & check on cacheIdentifier
  unsigned long long l1GtMenuCacheID = iSetup.get<L1TUtmTriggerMenuRcd>().cacheIdentifier();

  if (m_l1GtMenuCacheID != l1GtMenuCacheID) {
    edm::ESHandle<L1TUtmTriggerMenu> l1GtMenu;
    iSetup.get<L1TUtmTriggerMenuRcd>().get(l1GtMenu);
    const L1TUtmTriggerMenu* utml1GtMenu = l1GtMenu.product();

    // Instantiate Parser
    TriggerMenuParser gtParser = TriggerMenuParser();

    std::map<std::string, unsigned int> extBitMap = gtParser.getExternalSignals(utml1GtMenu);

    m_l1GtMenuCacheID = l1GtMenuCacheID;
    m_extBitMap = extBitMap;
  }

  bool foundBptxAND = (m_extBitMap.find("BPTX_plus_AND_minus.v0") != m_extBitMap.end());
  bool foundBptxPlus = (m_extBitMap.find("BPTX_plus.v0") != m_extBitMap.end());
  bool foundBptxMinus = (m_extBitMap.find("BPTX_minus.v0") != m_extBitMap.end());
  bool foundBptxOR = (m_extBitMap.find("BPTX_plus_OR_minus.v0") != m_extBitMap.end());

  unsigned int bitBptxAND = m_extBitMap["BPTX_plus_AND_minus.v0"];
  unsigned int bitBptxPlus = m_extBitMap["BPTX_plus.v0"];
  unsigned int bitBptxMinus = m_extBitMap["BPTX_minus.v0"];
  unsigned int bitBptxOR = m_extBitMap["BPTX_plus_OR_minus.v0"];

  edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
  iEvent.getByToken(gtReadoutRecordToken, gtReadoutRecord);

  // Setup vectors
  GlobalExtBlk extCond_bx_m2;
  GlobalExtBlk extCond_bx_m1;
  GlobalExtBlk extCond_bx_0;
  GlobalExtBlk extCond_bx_p1;
  GlobalExtBlk extCond_bx_p2;

  if (gtReadoutRecord.isValid()) {
    // L1GlobalTriggerReadoutRecord const & l1tResults = * gtReadoutRecord;

    // // select PSB#9 and bunch crossing 0
    // const L1GtPsbWord & psb = l1tResults.gtPsbWord(0xbb09, 0);

    // // the four 16-bit words psb.bData(1), psb.aData(1), psb.bData(0) and psb.aData(0) yield
    // // (in this sequence) the 64 technical trigger bits from most significant to least significant bit
    // uint64_t psbTriggerWord = ( (uint64_t) psb.bData(1) << 48) |
    // 	((uint64_t) psb.aData(1) << 32) |
    // 	((uint64_t) psb.bData(0) << 16) |
    // 	((uint64_t) psb.aData(0));

    // std::cout << "psbTriggerWord = " << psbTriggerWord << std::endl;
    // //

    for (int ibx = 0; ibx < 5; ibx++) {
      int useBx = ibx - 2;
      if (useBx < bxFirst_ || useBx > bxLast_)
        continue;

      //std::cout << "  BX = " << ibx - 2 << std::endl;

      // L1 technical
      const TechnicalTriggerWord& gtTTWord = gtReadoutRecord->technicalTriggerWord(useBx);
      int tbitNumber = 0;
      TechnicalTriggerWord::const_iterator GTtbitItr;
      bool passBptxAND = false;
      bool passBptxPlus = false;
      bool passBptxMinus = false;
      bool passBptxOR = false;
      for (GTtbitItr = gtTTWord.begin(); GTtbitItr != gtTTWord.end(); GTtbitItr++) {
        int pass_l1t_tech = 0;
        if (*GTtbitItr)
          pass_l1t_tech = 1;

        if (pass_l1t_tech == 1) {
          if (tbitNumber == 0)
            passBptxAND = true;
          else if (tbitNumber == 1)
            passBptxPlus = true;
          else if (tbitNumber == 2)
            passBptxMinus = true;
          else if (tbitNumber == 3)
            passBptxOR = true;
        }

        tbitNumber++;
      }

      if (useBx == -2) {
        if (passBptxAND && foundBptxAND)
          extCond_bx_m2.setExternalDecision(bitBptxAND, true);
        if (passBptxPlus && foundBptxPlus)
          extCond_bx_m2.setExternalDecision(bitBptxPlus, true);
        if (passBptxMinus && foundBptxMinus)
          extCond_bx_m2.setExternalDecision(bitBptxMinus, true);
        if (passBptxOR && foundBptxOR)
          extCond_bx_m2.setExternalDecision(bitBptxOR, true);
      } else if (useBx == -1) {
        if (passBptxAND && foundBptxAND)
          extCond_bx_m1.setExternalDecision(bitBptxAND, true);
        if (passBptxPlus && foundBptxPlus)
          extCond_bx_m1.setExternalDecision(bitBptxPlus, true);
        if (passBptxMinus && foundBptxMinus)
          extCond_bx_m1.setExternalDecision(bitBptxMinus, true);
        if (passBptxOR && foundBptxOR)
          extCond_bx_m1.setExternalDecision(bitBptxOR, true);
      } else if (useBx == 0) {
        if (passBptxAND && foundBptxAND)
          extCond_bx_0.setExternalDecision(bitBptxAND, true);
        if (passBptxPlus && foundBptxPlus)
          extCond_bx_0.setExternalDecision(bitBptxPlus, true);
        if (passBptxMinus && foundBptxMinus)
          extCond_bx_0.setExternalDecision(bitBptxMinus, true);
        if (passBptxOR && foundBptxOR)
          extCond_bx_0.setExternalDecision(bitBptxOR, true);
      } else if (useBx == 1) {
        if (passBptxAND && foundBptxAND)
          extCond_bx_p1.setExternalDecision(bitBptxAND, true);
        if (passBptxPlus && foundBptxPlus)
          extCond_bx_p1.setExternalDecision(bitBptxPlus, true);
        if (passBptxMinus && foundBptxMinus)
          extCond_bx_p1.setExternalDecision(bitBptxMinus, true);
        if (passBptxOR && foundBptxOR)
          extCond_bx_p1.setExternalDecision(bitBptxOR, true);
      } else if (useBx == 2) {
        if (passBptxAND && foundBptxAND)
          extCond_bx_p2.setExternalDecision(bitBptxAND, true);
        if (passBptxPlus && foundBptxPlus)
          extCond_bx_p2.setExternalDecision(bitBptxPlus, true);
        if (passBptxMinus && foundBptxMinus)
          extCond_bx_p2.setExternalDecision(bitBptxMinus, true);
        if (passBptxOR && foundBptxOR)
          extCond_bx_p2.setExternalDecision(bitBptxOR, true);
      }
    }
  } else {
    LogWarning("MissingProduct") << "Input L1GlobalTriggerReadoutRecord collection not found\n";
  }

  //outputs
  std::unique_ptr<GlobalExtBlkBxCollection> extCond(new GlobalExtBlkBxCollection(0, bxFirst_, bxLast_));

  // Fill Externals
  if (-2 >= bxFirst_ && -2 <= bxLast_)
    extCond->push_back(-2, extCond_bx_m2);
  if (-1 >= bxFirst_ && -1 <= bxLast_)
    extCond->push_back(-1, extCond_bx_m1);
  if (0 >= bxFirst_ && 0 <= bxLast_)
    extCond->push_back(0, extCond_bx_0);
  if (1 >= bxFirst_ && 1 <= bxLast_)
    extCond->push_back(1, extCond_bx_p1);
  if (2 >= bxFirst_ && 2 <= bxLast_)
    extCond->push_back(2, extCond_bx_p2);

  iEvent.put(std::move(extCond));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void L1TExtCondLegacyToStage2::fillDescriptions(ConfigurationDescriptions& descriptions) {
  // l1GtExtCondLegacyToStage2
  edm::ParameterSetDescription desc;
  desc.add<int>("bxFirst", -2);
  desc.add<int>("bxLast", 2);
  desc.add<edm::InputTag>("LegacyGtReadoutRecord", edm::InputTag("unpackLegacyGtDigis"));
  descriptions.add("l1GtExtCondLegacyToStage2", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TExtCondLegacyToStage2);
