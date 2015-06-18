// -*- C++ -*-
//
// Package:    HLTL1NumberFilter
// Class:      HLTL1NumberFilter
// 
/**\class HLTL1NumberFilter HLTL1NumberFilter.cc filter/HLTL1NumberFilter/src/HLTL1NumberFilter.cc

Description: 

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Martin Grunewald
//         Created:  Tue Jan 22 13:55:00 CET 2008
//
//


// system include files
#include <string>
#include <iostream>
#include <memory>

// user include files
#include "HLTrigger/special/interface/HLTL1NumberFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTL1NumberFilter::HLTL1NumberFilter(const edm::ParameterSet& config) :
  //now do what ever initialization is needed
  inputToken_( consumes<FEDRawDataCollection>(config.getParameter<edm::InputTag>("rawInput")) ),
  period_( config.getParameter<unsigned int>("period") ),
  fedId_(  config.getParameter<int>("fedId") ),
  invert_( config.getParameter<bool>("invert") )
{
}


HLTL1NumberFilter::~HLTL1NumberFilter()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


void
HLTL1NumberFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("rawInput",edm::InputTag("source"));
  desc.add<unsigned int>("period",4096);
  desc.add<bool>("invert",true);
  desc.add<int>("fedId",812);
  descriptions.add("hltL1NumberFilter",desc);
}
//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTL1NumberFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  using namespace edm;

  if (iEvent.isRealData()) {
    bool accept(false);
    edm::Handle<FEDRawDataCollection> theRaw ;
    iEvent.getByToken(inputToken_,theRaw) ;
    const FEDRawData& data = theRaw->FEDData(fedId_) ;
    if (data.data() && data.size() > 0){
      FEDHeader header(data.data()) ;
      if (period_!=0) accept = ( ( (header.lvl1ID())%period_ ) == 0 );
      if (invert_) accept = !accept;
      return accept;
    } else{
      LogWarning("HLTL1NumberFilter")<<"No valid data for FED "<<fedId_<<" used by HLTL1NumberFilter";
      return false;
    }
  } else {
    return true;
  }

}
