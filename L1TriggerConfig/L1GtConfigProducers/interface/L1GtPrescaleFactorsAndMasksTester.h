#ifndef L1GtConfigProducers_L1GtPrescaleFactorsAndMasksTester_h
#define L1GtConfigProducers_L1GtPrescaleFactorsAndMasksTester_h

/**
 * \class L1GtPrescaleFactorsAndMasksTester
 * 
 * 
 * Description: test analyzer for L1 GT prescale factors and masks.  
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

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations
// class declaration
class L1GtPrescaleFactorsAndMasksTester : public edm::EDAnalyzer
{

public:

    // constructor
    explicit L1GtPrescaleFactorsAndMasksTester(const edm::ParameterSet&);

    // destructor
    virtual ~L1GtPrescaleFactorsAndMasksTester();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

};

#endif /*L1GtConfigProducers_L1GtPrescaleFactorsAndMasksTester_h*/
