#ifndef L1GtConfigProducers_L1GtBoardMapsTester_h
#define L1GtConfigProducers_L1GtBoardMapsTester_h

/**
 * \class L1GtBoardMapsTester
 *
 *
 * Description: test analyzer for various mappings of the L1 GT boards.
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
class L1GtBoardMapsTester : public edm::EDAnalyzer {
public:
  // constructor
  explicit L1GtBoardMapsTester(const edm::ParameterSet&);

  // destructor
  ~L1GtBoardMapsTester() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
};

#endif /*L1GtConfigProducers_L1GtBoardMapsTester_h*/
