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
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1GtStableParameters;
class L1GtStableParametersRcd;

// class declaration
class L1GtStableParametersTester : public edm::global::EDAnalyzer<> {
public:
  // constructor
  explicit L1GtStableParametersTester(const edm::ParameterSet&);

  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::ESGetToken<L1GtStableParameters, L1GtStableParametersRcd> m_l1GtParToken;
};

#endif /*L1GtConfigProducers_L1GtStableParametersTester_h*/
