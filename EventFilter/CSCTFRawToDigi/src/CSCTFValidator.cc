/** 
 * Demo analyzer for reading digis.
 * Validates against raw data unpack.
 * author A.Tumanov 2/26/06 
 * ripped from Jeremy's and Rick's analyzers
 *   
 */
#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "EventFilter/CSCTFRawToDigi/interface/CSCTFValidator.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

CSCTFValidator::CSCTFValidator(edm::ParameterSet const& conf) {

  // If your module takes parameters, here is where you would define
  // their names and types, and access them to initialize internal
  // variables. Example as follows:
  //
  eventNumber = 0;
}

void CSCTFValidator::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {

  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
  edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
   
  // Pass the handle to the method "getByType", which is used to retrieve
  // one and only one instance of the type in question out of event "e". If
  // zero or more than one instance exists in the event an exception is thrown.
  //
  e.getByLabel("csctfunpacker","MuonCSCTFCorrelatedLCTDigi",corrlcts);
    
  // read digi collections and print digis
  //

  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator j=corrlcts->begin(); j!=corrlcts->end(); j++) {
 
    std::vector<CSCCorrelatedLCTDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCCorrelatedLCTDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }

  eventNumber++;
  edm::LogInfo ("CSCTFValidator")  << "end of event number " << eventNumber;
  
  

}
