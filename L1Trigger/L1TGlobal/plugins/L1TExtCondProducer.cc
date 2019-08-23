///
/// \class l1t::L1TExtCondProducer
///
/// Description: Fill uGT external condition to allow testing stage 2 algos, e.g. Bptx
///
///
/// \author: D. Puigh OSU
///

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
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "DataFormats/TCDS/interface/TCDSRecord.h"

using namespace std;
using namespace edm;
using namespace l1t;

//
// class declaration
//

class L1TExtCondProducer : public stream::EDProducer<> {
public:
  explicit L1TExtCondProducer(const ParameterSet&);
  ~L1TExtCondProducer() override;

  static void fillDescriptions(ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  // unsigned long long m_paramsCacheId; // Cache-ID from current parameters, to check if needs to be updated.
  //std::shared_ptr<const CaloParams> m_dbpars; // Database parameters for the trigger, to be updated as needed.
  //std::shared_ptr<const FirmwareVersion> m_fwv;
  //std::shared_ptr<FirmwareVersion> m_fwv; //not const during testing.

  // BX parameters
  int bxFirst_;
  int bxLast_;

  bool setBptxAND_;
  bool setBptxPlus_;
  bool setBptxMinus_;
  bool setBptxOR_;

  unsigned long long m_l1GtMenuCacheID;
  std::map<std::string, unsigned int> m_extBitMap;

  unsigned int m_triggerRulePrefireVetoBit;

  bool makeTriggerRulePrefireVetoBit_;
  edm::EDGetTokenT<TCDSRecord> tcdsRecordToken_;
  edm::InputTag tcdsInputTag_;
};

//
// constructors and destructor
//
L1TExtCondProducer::L1TExtCondProducer(const ParameterSet& iConfig)
    : bxFirst_(iConfig.getParameter<int>("bxFirst")),
      bxLast_(iConfig.getParameter<int>("bxLast")),
      setBptxAND_(iConfig.getParameter<bool>("setBptxAND")),
      setBptxPlus_(iConfig.getParameter<bool>("setBptxPlus")),
      setBptxMinus_(iConfig.getParameter<bool>("setBptxMinus")),
      setBptxOR_(iConfig.getParameter<bool>("setBptxOR")),
      tcdsInputTag_(iConfig.getParameter<edm::InputTag>("tcdsRecordLabel")) {
  makeTriggerRulePrefireVetoBit_ = false;

  m_triggerRulePrefireVetoBit = 255;
  if (m_triggerRulePrefireVetoBit > GlobalExtBlk::maxExternalConditions - 1) {
    m_triggerRulePrefireVetoBit = GlobalExtBlk::maxExternalConditions - 1;
    edm::LogWarning("L1TExtCondProducer")
        << "Default trigger rule prefire veto bit number too large. Resetting to " << m_triggerRulePrefireVetoBit;
  }

  if (!(tcdsInputTag_ == edm::InputTag(""))) {
    tcdsRecordToken_ = consumes<TCDSRecord>(tcdsInputTag_);
    makeTriggerRulePrefireVetoBit_ = true;
  }

  // register what you produce
  produces<GlobalExtBlkBxCollection>();

  // Initialize parameters
  m_l1GtMenuCacheID = 0ULL;
}

L1TExtCondProducer::~L1TExtCondProducer() {}

//
// member functions
//

// ------------ method called to produce the data ------------
void L1TExtCondProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  LogDebug("L1TExtCondProducer") << "L1TExtCondProducer::produce function called...\n";

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

  bool TriggerRulePrefireVetoBit(false);
  if (makeTriggerRulePrefireVetoBit_) {
    // code taken from Nick Smith's EventFilter/L1TRawToDigi/plugins/TriggerRulePrefireVetoFilter.cc

    edm::Handle<TCDSRecord> tcdsRecordH;
    iEvent.getByToken(tcdsRecordToken_, tcdsRecordH);
    const auto& tcdsRecord = *tcdsRecordH.product();

    uint64_t thisEvent = (tcdsRecord.getBXID() - 1) + tcdsRecord.getOrbitNr() * 3564ull;

    std::vector<uint64_t> eventHistory;
    for (auto&& l1a : tcdsRecord.getFullL1aHistory()) {
      eventHistory.push_back(thisEvent - ((l1a.getBXID() - 1) + l1a.getOrbitNr() * 3564ull));
    }

    // should be 16 according to TCDSRecord.h, we only care about the last 4
    if (eventHistory.size() < 4) {
      edm::LogError("L1TExtCondProducer") << "Unexpectedly small L1A history from TCDSRecord";
    }

    // No more than 1 L1A in 3 BX
    if (eventHistory[0] < 3ull) {
      edm::LogError("L1TExtCondProducer") << "Found an L1A in an impossible location?! (1 in 3)";
    }

    if (eventHistory[0] == 3ull)
      TriggerRulePrefireVetoBit = true;

    // No more than 2 L1As in 25 BX
    if (eventHistory[0] < 25ull and eventHistory[1] < 25ull) {
      edm::LogError("L1TExtCondProducer") << "Found an L1A in an impossible location?! (2 in 25)";
    }
    if (eventHistory[0] < 25ull and eventHistory[1] == 25ull)
      TriggerRulePrefireVetoBit = true;

    // No more than 3 L1As in 100 BX
    if (eventHistory[0] < 100ull and eventHistory[1] < 100ull and eventHistory[2] < 100ull) {
      edm::LogError("L1TExtCondProducer") << "Found an L1A in an impossible location?! (3 in 100)";
    }
    if (eventHistory[0] < 100ull and eventHistory[1] < 100ull and eventHistory[2] == 100ull)
      TriggerRulePrefireVetoBit = true;

    // No more than 4 L1As in 240 BX
    if (eventHistory[0] < 240ull and eventHistory[1] < 240ull and eventHistory[2] < 240ull and
        eventHistory[3] < 240ull) {
      edm::LogError("L1TExtCondProducer") << "Found an L1A in an impossible location?! (4 in 240)";
    }
    if (eventHistory[0] < 240ull and eventHistory[1] < 240ull and eventHistory[2] < 240ull and
        eventHistory[3] == 240ull)
      TriggerRulePrefireVetoBit = true;
  }

  // Setup vectors
  GlobalExtBlk extCond_bx;

  //outputs
  std::unique_ptr<GlobalExtBlkBxCollection> extCond(new GlobalExtBlkBxCollection(0, bxFirst_, bxLast_));

  bool foundBptxAND = (m_extBitMap.find("BPTX_plus_AND_minus.v0") != m_extBitMap.end());
  bool foundBptxPlus = (m_extBitMap.find("BPTX_plus.v0") != m_extBitMap.end());
  bool foundBptxMinus = (m_extBitMap.find("BPTX_minus.v0") != m_extBitMap.end());
  bool foundBptxOR = (m_extBitMap.find("BPTX_plus_OR_minus.v0") != m_extBitMap.end());

  // Fill in some external conditions for testing
  if (setBptxAND_ && foundBptxAND)
    extCond_bx.setExternalDecision(m_extBitMap["BPTX_plus_AND_minus.v0"], true);
  if (setBptxPlus_ && foundBptxPlus)
    extCond_bx.setExternalDecision(m_extBitMap["BPTX_plus.v0"], true);
  if (setBptxMinus_ && foundBptxMinus)
    extCond_bx.setExternalDecision(m_extBitMap["BPTX_minus.v0"], true);
  if (setBptxOR_ && foundBptxOR)
    extCond_bx.setExternalDecision(m_extBitMap["BPTX_plus_OR_minus.v0"], true);

  //check for updated Bptx names as well
  foundBptxAND = (m_extBitMap.find("ZeroBias_BPTX_AND_VME") != m_extBitMap.end());
  foundBptxPlus = (m_extBitMap.find("BPTX_B1_VME") != m_extBitMap.end());
  foundBptxMinus = (m_extBitMap.find("BPTX_B2_VME") != m_extBitMap.end());
  foundBptxOR = (m_extBitMap.find("BPTX_OR_VME") != m_extBitMap.end());

  // Fill in some external conditions for testing
  if (setBptxAND_ && foundBptxAND)
    extCond_bx.setExternalDecision(m_extBitMap["ZeroBias_BPTX_AND_VME"], true);
  if (setBptxPlus_ && foundBptxPlus)
    extCond_bx.setExternalDecision(m_extBitMap["BPTX_B1_VME"], true);
  if (setBptxMinus_ && foundBptxMinus)
    extCond_bx.setExternalDecision(m_extBitMap["BPTX_B2_VME"], true);
  if (setBptxOR_ && foundBptxOR)
    extCond_bx.setExternalDecision(m_extBitMap["BPTX_OR_VME"], true);

  // set the bit for the TriggerRulePrefireVeto if true
  if (TriggerRulePrefireVetoBit)
    extCond_bx.setExternalDecision(m_triggerRulePrefireVetoBit, true);

  // Fill Externals
  for (int iBx = bxFirst_; iBx <= bxLast_; iBx++) {
    extCond->push_back(iBx, extCond_bx);
  }

  iEvent.put(std::move(extCond));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void L1TExtCondProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  // simGtExtFakeProd
  edm::ParameterSetDescription desc;
  desc.add<bool>("setBptxMinus", true);
  desc.add<bool>("setBptxAND", true);
  desc.add<int>("bxFirst", -2);
  desc.add<bool>("setBptxOR", true);
  desc.add<int>("bxLast", 2);
  desc.add<bool>("setBptxPlus", true);
  desc.add<edm::InputTag>("tcdsRecordLabel", edm::InputTag(""));
  descriptions.add("simGtExtFakeProd", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TExtCondProducer);
