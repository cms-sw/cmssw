#ifndef HLTfilters_HLTLevel1GTSeed_h
#define HLTfilters_HLTLevel1GTSeed_h

/**
 * \class HLTLevel1GTSeed
 *
 *
 * Description: filter L1 bits and extract seed objects from L1 GT for HLT algorithms.
 *
 * Implementation:
 *    This class is an HLTStreamFilter (-> stream::EDFilter). It implements:
 *      - filtering on Level-1 bits, given via a logical expression of algorithm names
 *      - extraction of the seed objects from L1 GT object map record
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
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
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

#include "FWCore/Utilities/interface/InputTag.h"

// forward declarations
class L1GtTriggerMenu;
class L1GtTriggerMask;
class L1GlobalTriggerReadoutRecord;
class L1GlobalTriggerObjectMapRecord;
namespace edm {
  class ConfigurationDescriptions;
}

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

// class declaration
class HLTLevel1GTSeed : public HLTStreamFilter
{

public:

    /// constructor
    explicit HLTLevel1GTSeed(const edm::ParameterSet&);

    /// destructor
    virtual ~HLTLevel1GTSeed();

    /// parameter description
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    /// filter the event
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) override;

private:

    /// get the vector of object types for a condition cndName on the GTL chip chipNumber
    const std::vector<L1GtObject>* objectTypeVec(const int chipNumber, const std::string& cndName) const;

    /// update the tokenNumber (holding the bit numbers) from m_l1AlgoLogicParser
    /// for a new L1 Trigger menu
    void updateAlgoLogicParser(const L1GtTriggerMenu*, const AlgorithmMap&);

    /// update the tokenResult members from m_l1AlgoLogicParser
    /// for a new event
    void updateAlgoLogicParser(const std::vector<bool>& gtWord,
            const std::vector<unsigned int>& triggerMask, const int physicsDaqPartition);

    /// for seeding via technical triggers, convert the "name" to tokenNumber
    /// (seeding via bit numbers) - done once in constructor
    void convertStringToBitNumber();

    /// debug print grouped in a single function
    /// can be called for a new menu (bool "true") or for a new event
    void debugPrint(bool) const;

    /// seeding is done via L1 trigger object maps, considering the objects which fired in L1
    bool seedsL1TriggerObjectMaps(
            edm::Event &,
            trigger::TriggerFilterObjectWithRefs &,
            const L1GtTriggerMask *,
            const L1GlobalTriggerReadoutRecord *,
            const int physicsDaqPartition);

    /// seeding is done ignoring if a L1 object fired or not
    /// if the event is selected at L1, fill all the L1 objects of types corresponding to the
    /// L1 conditions from the seeding logical expression for bunch crosses F, 0, 1
    /// directly from L1Extra and use them as seeds at HLT
    /// method and filter return true if at least an object is filled
    bool seedsL1Extra(edm::Event &, trigger::TriggerFilterObjectWithRefs &) const;

    /// detailed print of filter content
    void dumpTriggerFilterObjectWithRefs(trigger::TriggerFilterObjectWithRefs &) const;


private:

    // cached stuff

    /// trigger menu
    const L1GtTriggerMenu * m_l1GtMenu;
    unsigned long long m_l1GtMenuCacheID;

    /// logic parser for m_l1SeedsLogicalExpression
    L1GtLogicParser m_l1AlgoLogicParser;

    /// list of required algorithms for seeding
    std::vector<L1GtLogicParser::OperandToken> m_l1AlgoSeeds;

    /// vector of Rpn vectors for the required algorithms for seeding
    std::vector< const std::vector<L1GtLogicParser::TokenRPN>* > m_l1AlgoSeedsRpn;

    /// vector of object-type vectors for each condition in the required algorithms for seeding
    std::vector< std::vector< const std::vector<L1GtObject>* > > m_l1AlgoSeedsObjType;


private:

    /// if true:
    ///    seeding done via L1 trigger object maps, with objects that fired
    ///    only objects from the central BxInEvent (L1A) are used
    /// if false:
    ///    seeding is done ignoring if a L1 object fired or not,
    ///    adding all L1EXtra objects corresponding to the object types
    ///    used in all conditions from the algorithms in logical expression
    ///    for a given number of BxInEvent
    bool m_l1UseL1TriggerObjectMaps;

    /// option used forL1UseL1TriggerObjectMaps = False only
    /// number of BxInEvent: 1: L1A=0; 3: -1, L1A=0, 1; 5: -2, -1, L1A=0, 1, 2
    int m_l1NrBxInEvent;

    /// seeding done via technical trigger bits, if value is "true"
    bool m_l1TechTriggerSeeding;

    /// seeding uses algorithm aliases instead of algorithm names, if value is "true"
    bool m_l1UseAliasesForSeeding;

    /// logical expression for the required L1 algorithms
    /// the algorithms are specified by name
    std::string m_l1SeedsLogicalExpression;

    /// InputTag for the L1 Global Trigger DAQ readout record
    edm::InputTag                                  m_l1GtReadoutRecordTag;
    edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1GtReadoutRecordToken;

    /// InputTag for L1 Global Trigger object maps
    edm::InputTag                                    m_l1GtObjectMapTag;
    edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> m_l1GtObjectMapToken;

    /// Meta InputTag for L1 particle collections (except muon)
    edm::InputTag m_l1CollectionsTag;

    /// Meta InputTag for L1 muon collection
    edm::InputTag m_l1MuonCollectionTag;

    /// cached InputTags
    edm::InputTag                                         m_l1ExtraTag;
    edm::InputTag                                         m_l1MuonTag;
    edm::EDGetTokenT<l1extra::L1MuonParticleCollection>   m_l1MuonToken;
    edm::InputTag                                         m_l1IsoEGTag;
    edm::EDGetTokenT<l1extra::L1EmParticleCollection>     m_l1IsoEGToken;
    edm::InputTag                                         m_l1NoIsoEGTag;
    edm::EDGetTokenT<l1extra::L1EmParticleCollection>     m_l1NoIsoEGToken;
    edm::InputTag                                         m_l1CenJetTag;
    edm::EDGetTokenT<l1extra::L1JetParticleCollection>    m_l1CenJetToken;
    edm::InputTag                                         m_l1ForJetTag;
    edm::EDGetTokenT<l1extra::L1JetParticleCollection>    m_l1ForJetToken;
    edm::InputTag                                         m_l1TauJetTag;
    edm::EDGetTokenT<l1extra::L1JetParticleCollection>    m_l1TauJetToken;
    edm::InputTag                                         m_l1IsoTauJetTag;
    edm::EDGetTokenT<l1extra::L1JetParticleCollection>    m_l1IsoTauJetToken;
    edm::InputTag                                         m_l1EtMissMETTag;
    edm::EDGetTokenT<l1extra::L1EtMissParticleCollection> m_l1EtMissMETToken;
    edm::InputTag                                         m_l1EtMissMHTTag;
    edm::EDGetTokenT<l1extra::L1EtMissParticleCollection> m_l1EtMissMHTToken;

    /// replace string "L1GlobalDecision" with bool to speed up the "if"
    bool m_l1GlobalDecision;

    /// cache edm::isDebugEnabled()
    bool m_isDebugEnabled;
};

#endif // HLTfilters_HLTLevel1GTSeed_h
