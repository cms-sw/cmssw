#ifndef L1GtConfigProducers_L1GtPsbSetupTester_h
#define L1GtConfigProducers_L1GtPsbSetupTester_h

/**
 * \class L1GtPsbSetupTester
 *
 *
 * Description: test analyzer for the setup of L1 GT PSB boards.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1GtPsbSetup;
class L1GtPsbSetupRcd;

// class declaration
class L1GtPsbSetupTester : public edm::global::EDAnalyzer<> {
public:
  // constructor
  explicit L1GtPsbSetupTester(const edm::ParameterSet&);

  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

private:
  edm::ESGetToken<L1GtPsbSetup, L1GtPsbSetupRcd> m_getToken;
};

#endif /*L1GtConfigProducers_L1GtPsbSetupTester_h*/
