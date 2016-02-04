#ifndef L1GtConfigProducers_L1GctConfigDump_h
#define L1GtConfigProducers_L1GctConfigDump_h

/**
 * \class L1GctConfigDump
 * 
 * 
 * Description: test analyzer for L1 GCT parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Jim Brooke
 * 
 * $Date: 2009/05/07 10:31:29 $
 * $Revision: 1.2 $
 *
 */


#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations


// class declaration
class L1GctConfigDump : public edm::EDAnalyzer
{

public:

    // constructor
    explicit L1GctConfigDump(const edm::ParameterSet&);

    // destructor
    virtual ~L1GctConfigDump();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    
};

#endif
