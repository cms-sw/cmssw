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
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations

// class declaration
class L1GtPsbSetupTester : public edm::EDAnalyzer {
public:
  // constructor
  explicit L1GtPsbSetupTester(const edm::ParameterSet&);

  // destructor
  ~L1GtPsbSetupTester() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
};

#endif /*L1GtConfigProducers_L1GtPsbSetupTester_h*/
