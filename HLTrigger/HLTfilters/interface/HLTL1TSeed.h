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
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"

// forward declarations
class L1GlobalTriggerObjectMapRecord;
namespace edm {
  class ConfigurationDescriptions;
}

// class declaration
class HLTL1TSeed : public HLTStreamFilter
{

public:

    /// constructor
    explicit HLTL1TSeed(const edm::ParameterSet&);

    /// destructor
    virtual ~HLTL1TSeed();

    /// parameter description
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    /// filter the event
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) override;

private:

    /// debug print grouped in a single function
    /// can be called for a new menu (bool "true") or for a new event
    void debugPrint(bool) const;

    /// seeding is done ignoring if a L1 object fired or not
    /// if the event is selected at L1, fill all the L1 objects of types corresponding to the
    /// L1 conditions from the seeding logical expression for bunch crosses F, 0, 1
    /// directly from L1Extra and use them as seeds at HLT
    /// method and filter return true if at least an object is filled
    bool seedsAll(edm::Event &, trigger::TriggerFilterObjectWithRefs &) const;

    /// detailed print of filter content
    void dumpTriggerFilterObjectWithRefs(trigger::TriggerFilterObjectWithRefs &) const;


private:
    // PRESENT IMPLEMENTATION IGNORES THIS, OPERATES AS IF SET TO FALSE
    /// if true:
    ///    seeding done via L1 trigger object maps, with objects that fired
    ///    only objects from the central BxInEvent (L1A) are used
    /// if false:
    ///    seeding is done ignoring if a L1 object fired or not,
    ///    adding all L1EXtra objects corresponding to the object types
    ///    used in all conditions from the algorithms in logical expression
    ///    for a given number of BxInEvent
    //bool useObjectMaps_;

    /// logical expression for the required L1 algorithms
    /// the algorithms are specified by name
    //std::string logicalExpression_;

    /// Meta InputTag for L1 Calo collections
    //edm::InputTag caloCollectionsTag_;

    /// Meta InputTag for L1 Muon collection
    edm::InputTag muonCollectionsTag_;
    edm::InputTag muonTag_;
    edm::EDGetTokenT<l1t::MuonBxCollection>   muonToken_;


    /// Meta InputTag for L1 Egamma collection
    edm::InputTag egammaCollectionsTag_;
    edm::InputTag egammaTag_;
    edm::EDGetTokenT<l1t::EGammaBxCollection>   egammaToken_;

    /// Meta InputTag for L1 Global collections
    //edm::InputTag globalCollectionsTag_;

    /// seeding uses algorithm aliases instead of algorithm names, if value is "true"
    //bool m_l1UseAliasesForSeeding;

    /// InputTag for L1 Global Trigger object maps
    //edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> m_l1GtObjectMapToken;


    /// cache edm::isDebugEnabled()
    bool m_isDebugEnabled;
};

#endif // HLTfilters_HLTL1TSeed_h
