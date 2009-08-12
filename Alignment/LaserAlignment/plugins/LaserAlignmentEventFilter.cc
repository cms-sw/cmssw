// Purpose: filter only needed events for LAS
// Original Author:  Adrian Perieanu
//         Created:  11th of August

#include "Alignment/LaserAlignment/plugins/LaserAlignmentEventFilter.h"

// constructor
LaserAlignmentEventFilter::LaserAlignmentEventFilter( const edm::ParameterSet& iConfig ):
  runFirst( iConfig.getUntrackedParameter<int>( "RunFirst", 0)),
  runLast( iConfig.getUntrackedParameter<int>( "RunLast", 0)),
  eventFirst( iConfig.getUntrackedParameter<int>( "EventFirst", 0)),
  eventLast( iConfig.getUntrackedParameter<int>( "EventLast", 0)) {
}
// destructor
LaserAlignmentEventFilter::~LaserAlignmentEventFilter() {
}

// filter
bool LaserAlignmentEventFilter::filter( edm::Event& iEvent, 
					const edm::EventSetup& iSetup) {
  bool selectEvent = false;
  int eventNum     = iEvent.id().event();
  int runNum       = iEvent.run();
  // select Run and Event range
  if( runFirst == runLast){
    if( ( runNum   == runFirst) &&
	( eventNum >= eventFirst && eventNum <= eventLast)){
      selectEvent = true;
      std::cout<<"LAS selectEvent."<<std::endl;
    }else{
      selectEvent = false;
    }
  }else{
    if( ( runNum >= runFirst && eventNum >= eventFirst) &&
	( runNum <= runLast  && eventNum <= eventLast)){
      selectEvent = true;
      std::cout<<"LAS selectEvent."<<std::endl;
    }else{
      selectEvent = false;
    }
  }
  return selectEvent;
}

void LaserAlignmentEventFilter::beginJob( const edm::EventSetup& ) {
}

void LaserAlignmentEventFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(LaserAlignmentEventFilter);
