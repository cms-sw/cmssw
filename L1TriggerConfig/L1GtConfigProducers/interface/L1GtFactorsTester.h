#ifndef L1GtConfigProducers_L1GtFactorsTester_h
#define L1GtConfigProducers_L1GtFactorsTester_h

/**
 * \class L1GtFactorsTester
 * 
 * 
 * Description: test analyzer for L1 GT prescale factors and masks.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtFactorsTester.h"

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations
class L1GtPrescaleFactors;
class L1GtTriggerMask;

// class declaration
class L1GtFactorsTester : public edm::EDAnalyzer
{

public:

    // constructor
    explicit L1GtFactorsTester(const edm::ParameterSet&);

    // destructor
    virtual ~L1GtFactorsTester();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

};

#endif /*L1GtConfigProducers_L1GtFactorsTester_h*/
