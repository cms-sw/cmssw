/** 
 * Demo analyzer for reading digis
 * author A.Tumanov 2/22/06 
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
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "EventFilter/CSCRawToDigi/interface/DigiAnalyzer.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"

DigiAnalyzer::DigiAnalyzer(edm::ParameterSet const& conf) {

  // If your module takes parameters, here is where you would define
  // their names and types, and access them to initialize internal
  // variables. Example as follows:
  //
  eventNumber = 0;
}

void DigiAnalyzer::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {

  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
  edm::Handle<CSCWireDigiCollection> wires;
 
  // Pass the handle to the method "getByType", which is used to retrieve
  // one and only one instance of the type in question out of event "e". If
  // zero or more than one instance exists in the event an exception is thrown.
  //
  e.getByLabel("cscunpacker","MuonCSCWireDigi",wires);
 
  
   
  // "do stuff"
  //
  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      digiItr->print();
    }
  }
  
  eventNumber++;
  std::cout << "event number " << eventNumber << std::endl;
  


}





