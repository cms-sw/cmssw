#ifndef L1Trigger_GlobalTrigger_CompareToObjectMapRecord_h
#define L1Trigger_GlobalTrigger_CompareToObjectMapRecord_h

/**
 * \class CompareToObjectMapRecord
 * 
 * 
 * Description:Compares the L1GlobalTriggerObjectMapRecord
 *             to the L1GlobalTriggerObjectMaps object and
 *             also the ParameterSet registry and verifies that
 *             the information is consistent or else it
 *             throws.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * \author: W. David Dagenhart
 * 
 * $Date: 2012/03/02 22:03:25 $
 * $Revision: 1.1 $
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

class CompareToObjectMapRecord : public edm::EDAnalyzer {

public:
    explicit CompareToObjectMapRecord(const edm::ParameterSet& pset);
    ~CompareToObjectMapRecord();

    virtual void analyze(edm::Event const& event, edm::EventSetup const& es);

private:
    edm::InputTag m_l1GtObjectMapTag;
    edm::InputTag m_l1GtObjectMapsTag;
    bool verbose_;
};

#endif
