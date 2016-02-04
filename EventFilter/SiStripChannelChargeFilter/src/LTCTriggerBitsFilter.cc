// -*- C++ -*-
//
// Package:     SiStripChannelChargeFilter
// Class  :     LTCTriggerBitsFilter
// 
//
// Original Author:  dkcira

#include "EventFilter/SiStripChannelChargeFilter/interface/LTCTriggerBitsFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include <iostream>

namespace cms
{

LTCTriggerBitsFilter::LTCTriggerBitsFilter(const edm::ParameterSet& ps){
   AcceptFromBits = ps.getParameter< std::vector<unsigned> >("AcceptFromBits");
   // check if bits > 7
   for( std::vector<uint32_t>::iterator ibit = AcceptFromBits.begin(); ibit != AcceptFromBits.end(); ibit++){
     if( *ibit > 7U ) { // *ibit >= 0, since *ibit is unsigned
      edm::LogInfo("LTCTriggerBitsFilter")<< *ibit<<" is not a valid bit, it will be ignored. Bits can go from 0 to 7";
      AcceptFromBits.erase(ibit);
      --ibit; // ! go back one as the nr. of elements changed. this way make sure no element is skipped
     }
   }
   //
   edm::LogInfo("LTCTriggerBitsFilter")<<"Will accept events from "<<AcceptFromBits.size()<<" LTC bits:";
   std::cout<<"LTCTriggerBitsFilter"<<"Will accept events from "<<AcceptFromBits.size()<<" LTC bits:"<<std::endl;
   for( std::vector<uint32_t>::const_iterator ibit = AcceptFromBits.begin(); ibit != AcceptFromBits.end(); ibit++){
          edm::LogInfo("LTCTriggerBitsFilter")<< *ibit;
          std::cout<<"LTCTriggerBitsFilter "<< *ibit<<std::endl;
   }
   produces <int>();
}

  // 0 DT
  // 1 CSC
  // 2 RBC1 (RPC techn. cosmic trigger for wheel +1, sector 10)
  // 3 RBC2 (RPC techn. cosmic trigger for wheel +2, sector 10)
  // 4 RPCTB (RPC Trigger Board trigger, covering both sectors 10 of both wheels, but with different geometrical acceptance ("pointing"))
  // 5 unused 

bool LTCTriggerBitsFilter::filter(edm::Event & e, edm::EventSetup const& c) {
  bool decision=false; // default value, only accept if set true in this loop
  if(AcceptFromBits.size()!=0){
    edm::Handle<LTCDigiCollection> ltcdigis;
    e.getByType(ltcdigis);
    for( LTCDigiCollection::const_iterator ltcdigiItr = ltcdigis->begin() ; ltcdigiItr != ltcdigis->end() ; ++ltcdigiItr ) {
      for( std::vector<uint32_t>::const_iterator jbit = AcceptFromBits.begin(); jbit != AcceptFromBits.end(); jbit++){
        if(ltcdigiItr->HasTriggered(*jbit)) decision = true; // keep the event if any of requested triggers fired
      }
    }
  }
  std::auto_ptr< int > output_decision( new int(decision) );
  e.put(output_decision);
  return decision;
}

}
