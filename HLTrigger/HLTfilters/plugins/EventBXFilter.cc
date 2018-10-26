#include <iostream>

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "EventBXFilter.h"

//
// constructors and destructor
//
EventBXFilter::EventBXFilter(const edm::ParameterSet& iConfig)
  : allowedBXs_ ( iConfig.getParameter<std::vector<unsigned int> >("allowedBXs") )
  , vetoBXs_    ( iConfig.getParameter<std::vector<unsigned int> >("vetoBXs")    )
{
  for ( auto aBX : allowedBXs_ )
    if ( std::find(vetoBXs_.begin(), vetoBXs_.end(), aBX) != vetoBXs_.end() )
      edm::LogWarning("EventBXFilter") << "same BX index (" << aBX << ") appears in both allowed and veto lists" << std::endl;
}

EventBXFilter::~EventBXFilter()
{}

//
// member functions
//

void
EventBXFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<unsigned int> allowedBXs; // # provide list of BX you want
  desc.add<std::vector<unsigned int> >("allowedBXs",allowedBXs);
  std::vector<unsigned int> vetoBXs; // # provide list of BX you do not want
  desc.add<std::vector<unsigned int> >("vetoBXs",vetoBXs);
  descriptions.add("eventBXFilter", desc);
}

// ------------ method called to produce the data  ------------
  bool
EventBXFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  bool accept = true; // if both allowedBXs and vetoBXs vectors are empty, the event pass
  unsigned int bx = iEvent.bunchCrossing();
  //  std::cout << iEvent.id().run() << " " << iEvent.id().event() << " " << iEvent.luminosityBlock() << " " << bx << std::endl;

  if ( !vetoBXs_.empty() )
    if ( std::find(vetoBXs_.begin(), vetoBXs_.end(), bx) != vetoBXs_.end() )
      accept = false;
    else
      accept = true;

  if ( !allowedBXs_.empty() )
    if ( std::find(allowedBXs_.begin(), allowedBXs_.end(), bx) != allowedBXs_.end() )
      accept = true;
    else
      accept = false;

  return accept;
}
