#ifndef L1GtConfigProducers_L1GtStableParametersTester_h
#define L1GtConfigProducers_L1GtStableParametersTester_h

/**
 * \class L1GtStableParametersTester
 * 
 * 
 * Description: test analyzer for L1 GT parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtStableParametersTester.h"

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1GtStableParameters;

// class declaration
class L1GtStableParametersTester : public edm::EDAnalyzer {
public:
  // constructor
  explicit L1GtStableParametersTester(const edm::ParameterSet&);

  // destructor
  ~L1GtStableParametersTester() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
};

#endif /*L1GtConfigProducers_L1GtStableParametersTester_h*/
