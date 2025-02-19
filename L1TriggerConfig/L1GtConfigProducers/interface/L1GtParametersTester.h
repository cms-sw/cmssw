#ifndef L1GtConfigProducers_L1GtParametersTester_h
#define L1GtConfigProducers_L1GtParametersTester_h

/**
 * \class L1GtParametersTester
 * 
 * 
 * Description: test analyzer for L1 GT parameters.  
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtParametersTester.h"

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations
class L1GtParameters;

// class declaration
class L1GtParametersTester : public edm::EDAnalyzer
{

public:

    // constructor
    explicit L1GtParametersTester(const edm::ParameterSet&);

    // destructor
    virtual ~L1GtParametersTester();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

};

#endif /*L1GtConfigProducers_L1GtParametersTester_h*/
