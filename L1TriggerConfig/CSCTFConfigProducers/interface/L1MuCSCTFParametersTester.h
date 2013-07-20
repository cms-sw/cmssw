#ifndef CSCTFConfigProducers_L1MuCSCTFParametersTester_h
#define CSCTFConfigProducers_L1MuCSCTFParametersTester_h

/**
 * \class L1MuCSCTFParametersTester
 * 
 * 
 * Description: test analyzer for L1 CSCTF parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: G.P. Di Giovanni - University of Florida
 * 
 * $Date: 2010/05/21 13:04:50 $
 * $Revision: 1.1 $
 *
 */

// this class header
#include "L1TriggerConfig/CSCTFConfigProducers/interface/L1MuCSCTFParametersTester.h"

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations
class L1GtParameters;

// class declaration
class L1MuCSCTFParametersTester : public edm::EDAnalyzer
{

public:

    // constructor
    explicit L1MuCSCTFParametersTester(const edm::ParameterSet&);

    // destructor
    virtual ~L1MuCSCTFParametersTester();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

};

#endif /*CSCTFConfigProducers_L1MuCSCTFParametersTester_h*/
