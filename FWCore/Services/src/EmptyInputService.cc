/*----------------------------------------------------------------------
$Id: EmptyInputService.cc,v 1.9 2005/07/30 04:27:03 wmtan Exp $
----------------------------------------------------------------------*/

#include <stdexcept>
#include <memory>
#include <string>

#include "FWCore/Services/src/EmptyInputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/InputServiceDescription.h"
#include "FWCore/Framework/interface/InputService.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/EDProduct/interface/EventID.h"

namespace edm {
  class BranchKey;
  FakeRetriever::~FakeRetriever() {}

  std::auto_ptr<EDProduct>
  FakeRetriever::get(BranchKey const&) {
    throw std::runtime_error("FakeRetriever::get called");
  }

  EmptyInputService::EmptyInputService(ParameterSet const& pset,
				       InputServiceDescription const& desc) :
    InputService(desc),
    nextID_(EventID::firstValidEvent() ),
    remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)),
    retriever_(new FakeRetriever())
  { }

  EmptyInputService::~EmptyInputService() {
    delete retriever_;
  }

  std::auto_ptr<EventPrincipal>
  EmptyInputService::read() {
    std::auto_ptr<EventPrincipal> result(0);
    
    if (remainingEvents_-- != 0) {
      result = std::auto_ptr<EventPrincipal>(new EventPrincipal(nextID_, *retriever_, *preg_));
      nextID_ = nextID_.next();
    }
    return result;
  }
}
