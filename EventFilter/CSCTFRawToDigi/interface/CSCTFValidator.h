#ifndef CSCTFRawToDigi_CSCTFValidator_h
#define CSCTFRawToDigi_CSCTFValidator_h

/** 
 * A basic analyzer used to validate Digis written into
 * the event by CSCTFUnpacker.
 * \author L. Gray 2/26/06 
 *   
 */

#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"


class CSCTFValidator : public edm::EDAnalyzer {
 public:
  explicit CSCTFValidator(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  //virtual void endJob();
 private:
  // variables persistent across events should be declared here.
  //
  int eventNumber;
};

#endif
