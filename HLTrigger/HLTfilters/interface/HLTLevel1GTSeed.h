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
 *      - filtering on Level-1 bits, given via a logical expression 
 *        of algorithm names or bit numbers
 *      - extraction of the seed objects from L1 GT object map record
 *    Initial implementation: compatible with HLTLevel1Seed (author: M. Gruenewald) 
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include<vector>
#include<string>

// user include files

//   base class
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"


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

    L1GtObject objectType(const int algoBit, const int indexCond, const int indexObj,
                          const edm::EventSetup&);

    /// get map from algorithm names to algorithm bits 
    void getAlgoMap(edm::Event&, const edm::EventSetup&); 
    
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
    
    // temporary, until L1 trigger menu implemented as EventSetup
    std::map<std::string, int> m_algoNameToBit;
    bool m_algoMap;

};

#endif // HLTfilters_HLTLevel1GTSeed_h
