#ifndef HLTfilters_HLTLevel1GTSeed_h
#define HLTfilters_HLTLevel1GTSeed_h

/**
 * \class HLTLevel1GTSeed
 * 
 * 
 * Description: filter L1 bits and extract seed objects from L1 GT for HLT algorithms.  
 *
 * Implementation:
 *    This class is an HLTFilter (-> EDFilter). It implements: 
 *      - filtering on Level-1 bits, given via a logical expression of algorithm names
 *      - extraction of the seed objects from L1 GT object map record
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>
#include <vector>

// user include files

//   base class
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

// forward declarations
class L1GtTriggerMenu;
class L1GtTriggerMask;

// class declaration
class HLTLevel1GTSeed : public HLTFilter
{

public:

    /// constructor
    explicit HLTLevel1GTSeed(const edm::ParameterSet&);

    /// destructor
    virtual ~HLTLevel1GTSeed();

    /// filter the event
    virtual bool filter(edm::Event&, const edm::EventSetup&);

private:

    /// get the vector of object types for a condition cndName on the GTL chip chipNumber
    const std::vector<L1GtObject>* objectTypeVec(const int chipNumber, const std::string& cndName);

    /// update the tokenNumber (holding the bit numbers) from m_l1AlgoLogicParser 
    /// for a new L1 Trigger menu
    void updateAlgoLogicParser(const L1GtTriggerMenu*);

    /// update the tokenResult members from m_l1AlgoLogicParser 
    /// for a new event
    void updateAlgoLogicParser(const std::vector<bool>& gtWord, 
            const std::vector<unsigned int>& triggerMask, const int physicsDaqPartition);
    
    /// for seeding via technical triggers, convert the "name" to tokenNumber
    /// (seeding via bit numbers)
    void convertStringToBitNumber();

    /// debug print grouped in a single function
    /// can be called for a new menu (bool "true") or for a new event
    void debugPrint(bool);
        

private:
    
    // cached stuff

    /// trigger menu
    const L1GtTriggerMenu* m_l1GtMenu;
    unsigned long long m_l1GtMenuCacheID;

    /// trigger masks
    const L1GtTriggerMask* m_l1GtTmAlgo;
    unsigned long long m_l1GtTmAlgoCacheID;
 
    const L1GtTriggerMask* m_l1GtTmTech;
    unsigned long long m_l1GtTmTechCacheID;
    
    std::vector<unsigned int> m_triggerMaskAlgoTrig;
    std::vector<unsigned int> m_triggerMaskTechTrig;

    //
    
    /// logic parser for m_l1SeedsLogicalExpression
    L1GtLogicParser m_l1AlgoLogicParser;

    /// list of required algorithms for seeding 
    std::vector<L1GtLogicParser::OperandToken> m_l1AlgoSeeds;
    
    /// vector of Rpn vectors for the required algorithms for seeding 
    std::vector< const std::vector<L1GtLogicParser::TokenRPN>* > m_l1AlgoSeedsRpn;

    /// vector of object-type vectors for each condition in the required algorithms for seeding 
    std::vector< std::vector< const std::vector<L1GtObject>* > > m_l1AlgoSeedsObjType;

private:

    /// seeding done via technical trigger bits, if value is "true";
    bool m_l1TechTriggerSeeding;

    /// logical expression for the required L1 algorithms;
    /// the algorithms are specified by name
    std::string m_l1SeedsLogicalExpression;

    /// InputTag for the L1 Global Trigger DAQ readout record
    edm::InputTag m_l1GtReadoutRecordTag;

    /// InputTag for L1 Global Trigger object maps
    edm::InputTag m_l1GtObjectMapTag;

    /// InputTag for L1 particle collections (except muon)    
    edm::InputTag m_l1CollectionsTag;

    /// InputTag for L1 muon collection
    edm::InputTag m_l1MuonCollectionTag;

    /// Whether to save these tags
    bool saveTags_;

};

#endif // HLTfilters_HLTLevel1GTSeed_h
