#ifndef L1GtConfigProducers_L1GctConfigDump_h
#define L1GtConfigProducers_L1GctConfigDump_h

/**
 * \class L1GctConfigDump
 * 
 * 
 * Description: test analyzer for L1 GCT parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Jim Brooke
 * 
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations

// class declaration
class L1GctConfigDump : public edm::EDAnalyzer {
public:
  // constructor
  explicit L1GctConfigDump(const edm::ParameterSet&);

  // destructor
  ~L1GctConfigDump() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
};

#endif
