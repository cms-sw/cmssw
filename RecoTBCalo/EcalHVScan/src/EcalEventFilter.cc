//
// filter module to run over events between firstEvent and lastEvent
// NB: the interval is inclusive! both firstEvent and lastEvent are selected OR rejected
// Author: Shahram Rahatlou, University of Rome & INFN
// Date:   16 Dec 2005
// $Id: EcalEventFilter.cc,v 1.2 2006/04/21 01:45:25 wmtan Exp $
//
#include "RecoTBCalo/EcalHVScan/src/EcalEventFilter.h"
#include <iostream>

EcalEventFilter::EcalEventFilter( const edm::ParameterSet& ps ) {

  firstEvt_ = ps.getUntrackedParameter<int>("firstEvt",-1);
  lastEvt_  = ps.getUntrackedParameter<int>("lastEvt",10000000);
  selectInBetween_  = ps.getUntrackedParameter<bool>("selectInBetween",true);

  std::cout << "EcalEventFilter: will select only events between event " << firstEvt_ << " and event " << lastEvt_ << std::endl;

  rejectedEvt_ = 0;

}


EcalEventFilter::~EcalEventFilter() {

}

void EcalEventFilter::endJob() {

  std::cout << "EcalEventFilter: rejected " << rejectedEvt_ << " events." << std::endl;

}

bool
EcalEventFilter::filter(edm::Event& evt, edm::EventSetup const& es) {

  int thisEvt = evt.id().event();

  bool result =   ( selectInBetween_ && (thisEvt>=firstEvt_) && (thisEvt<=lastEvt_) ) ||  // events between first and last
                 ( !selectInBetween_ && (thisEvt<firstEvt_  || thisEvt>lastEvt_)  );    // events before first and after last

  if(!result) rejectedEvt_++;
  return result;

}
