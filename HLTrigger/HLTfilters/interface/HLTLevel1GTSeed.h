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
#include <map>

// user include files

//   base class
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

// forward declarations

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

    L1GtObject objectType(const std::string& cndName, const int& indexObj,
        const std::vector<ConditionMap>& conditionMap);

    /// get map of (algorithm names, algorithm bits)
    std::map<std::string, int> mapAlgNameToBit(const AlgorithmMap&);

private:
    /// logical expression for the required L1 algorithms;
    /// the algorithms can be given by name or by bit number
    std::string m_l1SeedsLogicalExpression;

    /// InputTag for the L1 Global Trigger DAQ readout record
    edm::InputTag m_l1GtReadoutRecordTag;

    /// InputTag for L1 Global Trigger object maps
    edm::InputTag m_l1GtObjectMapTag;

    /// InputTag for L1 particle collections
    edm::InputTag m_l1CollectionsTag;
    edm::InputTag m_l1MuonCollectionTag;

};

#endif // HLTfilters_HLTLevel1GTSeed_h
