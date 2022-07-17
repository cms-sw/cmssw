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
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1GtBoardMaps;
class L1GtBoardMapsRcd;

// class declaration
class L1GtBoardMapsTester : public edm::global::EDAnalyzer<> {
public:
  // constructor
  explicit L1GtBoardMapsTester(const edm::ParameterSet&);

  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

private:
  edm::ESGetToken<L1GtBoardMaps, L1GtBoardMapsRcd> m_getToken;
};

#endif /*L1GtConfigProducers_L1GtBoardMapsTester_h*/
