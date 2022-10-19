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
 *
 */

// this class header
#include "L1TriggerConfig/CSCTFConfigProducers/interface/L1MuCSCTFParametersTester.h"

// system include files

// user include files
//   base class
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1MuCSCTFConfiguration;
class L1MuCSCTFConfigurationRcd;

// class declaration
class L1MuCSCTFParametersTester : public edm::one::EDAnalyzer<> {
public:
  // constructor
  explicit L1MuCSCTFParametersTester(const edm::ParameterSet&);

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::ESGetToken<L1MuCSCTFConfiguration, L1MuCSCTFConfigurationRcd> token_;
};

#endif /*CSCTFConfigProducers_L1MuCSCTFParametersTester_h*/
