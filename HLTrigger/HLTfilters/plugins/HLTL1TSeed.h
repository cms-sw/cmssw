#ifndef HLTfilters_HLTL1TSeed_h
#define HLTfilters_HLTL1TSeed_h

/**
 * \class HLTL1TSeed
 *
 *
 * Description: filter L1 bits and extract seed objects from L1 GT for HLT algorithms.
 *
 * Implementation:
 *    This class is an HLTStreamFilter (-> stream::EDFilter). It implements:
 *      - filtering on Level-1 bits, given via a logical expression of algorithm names (currently ignored)
 *      - extraction of the seed objects from L1T uGT object map record
 *
 */

// system include files
#include <string>
#include <vector>

// user include files

//   base class
#include "HLTrigger/HLTcore/interface/HLTStreamFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "DataFormats/L1TGlobal/interface/GlobalLogicParser.h"
#include "DataFormats/L1TGlobal/interface/GlobalObject.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"

#include "FWCore/Utilities/interface/InputTag.h"

// forward declarations
class L1GtTriggerMenu;
class L1GtTriggerMask;
class L1GlobalTriggerReadoutRecord;

class GlobalObjectMapRecord;
namespace edm {
  class ConfigurationDescriptions;
}

// class declaration
class HLTL1TSeed : public HLTStreamFilter {
public:
  /// constructor
  explicit HLTL1TSeed(const edm::ParameterSet&);

  /// destructor
  ~HLTL1TSeed() override;

  /// parameter description
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// filter the event
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs& filterproduct) override;

private:
  /// update the tokenNumber (holding the bit numbers) from m_l1AlgoLogicParser
  /// for a new L1 Trigger menu
  void inline updateAlgoLogicParser(const L1GtTriggerMenu*, const AlgorithmMap&){};

  /// update the tokenResult members from m_l1AlgoLogicParser
  /// for a new event
  void inline updateAlgoLogicParser(const std::vector<bool>& gtWord,
                                    const std::vector<unsigned int>& triggerMask,
                                    const int physicsDaqPartition){};

  /// seeding is done via L1 trigger object maps, considering the objects which fired in L1
  bool seedsL1TriggerObjectMaps(edm::Event&, trigger::TriggerFilterObjectWithRefs&);

  /// detailed print of filter content
  void dumpTriggerFilterObjectWithRefs(trigger::TriggerFilterObjectWithRefs&) const;

private:
  /// logic parser for m_l1SeedsLogicalExpression
  GlobalLogicParser m_l1AlgoLogicParser;

  /// list of required algorithms for seeding
  std::vector<GlobalLogicParser::OperandToken> m_l1AlgoSeeds;

  /// vector of Rpn vectors for the required algorithms for seeding
  std::vector<const std::vector<GlobalLogicParser::TokenRPN>*> m_l1AlgoSeedsRpn;

  /// vector of object-type vectors for each condition in the required algorithms for seeding
  std::vector<std::vector<const std::vector<l1t::GlobalObject>*> > m_l1AlgoSeedsObjType;

private:
  /// option used forL1UseL1TriggerObjectMaps = False only
  /// number of BxInEvent: 1: L1A=0; 3: -1, L1A=0, 1; 5: -2, -1, L1A=0, 1, 2
  int m_l1NrBxInEvent;

  /// logical expression for the required L1 algorithms
  /// the algorithms are specified by name
  std::string m_l1SeedsLogicalExpression;

  /// InputTag for L1 Global Trigger object maps. This is done per menu. Should be part of Run.
  edm::InputTag m_l1GtObjectMapTag;
  edm::EDGetTokenT<GlobalObjectMapRecord> m_l1GtObjectMapToken;

  /// InputTag for L1 Global Trigger
  edm::InputTag m_l1GlobalTag;
  edm::EDGetTokenT<GlobalAlgBlkBxCollection> m_l1GlobalToken;

  //edm::InputTag dummyTag;
  /// Meta InputTag for L1 Muon collection
  edm::InputTag m_l1MuonCollectionsTag;
  edm::InputTag m_l1MuonTag;
  edm::EDGetTokenT<l1t::MuonBxCollection> m_l1MuonToken;

  /// Meta InputTag for L1 Muon collection
  edm::InputTag m_l1MuonShowerCollectionsTag;
  edm::InputTag m_l1MuonShowerTag;
  edm::EDGetTokenT<l1t::MuonShowerBxCollection> m_l1MuonShowerToken;

  /// Meta InputTag for L1 Egamma collection
  edm::InputTag m_l1EGammaCollectionsTag;
  edm::InputTag m_l1EGammaTag;
  edm::EDGetTokenT<l1t::EGammaBxCollection> m_l1EGammaToken;

  /// Meta InputTag for L1 Jet collection
  edm::InputTag m_l1JetCollectionsTag;
  edm::InputTag m_l1JetTag;
  edm::EDGetTokenT<l1t::JetBxCollection> m_l1JetToken;

  /// Meta InputTag for L1 Tau collection
  edm::InputTag m_l1TauCollectionsTag;
  edm::InputTag m_l1TauTag;
  edm::EDGetTokenT<l1t::TauBxCollection> m_l1TauToken;

  /// Meta InputTag for L1 EtSum collection
  edm::InputTag m_l1EtSumCollectionsTag;
  edm::InputTag m_l1EtSumTag;
  edm::EDGetTokenT<l1t::EtSumBxCollection> m_l1EtSumToken;

  /// Meta InputTag for L1 EtSum (ZDC) collection
  edm::InputTag m_l1EtSumZdcCollectionsTag;
  edm::InputTag m_l1EtSumZdcTag;
  edm::EDGetTokenT<l1t::EtSumBxCollection> m_l1EtSumZdcToken;

  /// flag to pass if L1TGlobal accept
  bool m_l1GlobalDecision;

  /// cache edm::isDebugEnabled()
  bool m_isDebugEnabled;
};

#endif  // HLTfilters_HLTL1TSeed_h
