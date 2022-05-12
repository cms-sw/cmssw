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
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtParametersTester.h"

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1GtParameters;
class L1GtParametersRcd;

// class declaration
class L1GtParametersTester : public edm::global::EDAnalyzer<> {
public:
  // constructor
  explicit L1GtParametersTester(const edm::ParameterSet&);

  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::ESGetToken<L1GtParameters, L1GtParametersRcd> m_l1GtParToken;
};

#endif /*L1GtConfigProducers_L1GtParametersTester_h*/
