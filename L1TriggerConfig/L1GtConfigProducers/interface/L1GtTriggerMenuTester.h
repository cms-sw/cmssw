#ifndef L1GtConfigProducers_L1GtTriggerMenuTester_h
#define L1GtConfigProducers_L1GtTriggerMenuTester_h

/**
 * \class L1GtTriggerMenuTester
 * 
 * 
 * Description: test analyzer for L1 GT trigger menu.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuTester.h"

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations


// class declaration
class L1GtTriggerMenuTester : public edm::EDAnalyzer
{

public:

    // constructor
    explicit L1GtTriggerMenuTester(const edm::ParameterSet&);

    // destructor
    virtual ~L1GtTriggerMenuTester();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

};

#endif /*L1GtConfigProducers_L1GtTriggerMenuTester_h*/
