#include "FWCore/FWCoreServices/src/EventInfoPrinter.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include <iostream>
using namespace edm;
using namespace std;

EventInfoPrinter::EventInfoPrinter( const edm::ParameterSet & ) : 
  counter( 0 ) {
}

EventInfoPrinter::~EventInfoPrinter() { 
  cout << ">>> processed " << counter << " events" << endl;
}

void EventInfoPrinter::analyze( const edm::Event& evt, const edm::EventSetup& ) {
  CollisionID id = evt.id();
  //  const Run & run = evt.getRun(); // this is still unused
  cout << ">>> processing event # " << id << endl;
  counter ++;
}


