/*----------------------------------------------------------------------
$Id: EmptyInputService.cc,v 1.2 2005/06/02 20:18:46 wmtan Exp $
----------------------------------------------------------------------*/

#include <stdexcept>
#include <memory>
#include <string>

#include "FWCore/FWCoreServices/src/EmptyInputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/CoreFramework/interface/InputServiceDescription.h"
#include "FWCore/CoreFramework/interface/InputService.h"
#include "FWCore/CoreFramework/interface/EventPrincipal.h"
#include "FWCore/EDProduct/interface/CollisionID.h"

namespace edm {
  class BranchKey;
  FakeRetriever::~FakeRetriever() {}

  std::auto_ptr<EDProduct>
  FakeRetriever::get(BranchKey const&) {
    throw std::runtime_error("FakeRetriever::get called");
  }

  EmptyInputService::EmptyInputService(ParameterSet const& pset,
				       const InputServiceDescription& desc) :
    InputService(desc.process_name),
    nextID_(1),
    remainingEvents_(getUntrackedP<int>(pset, "maxEvents", -1)),
    retriever_(new FakeRetriever())
  { }

  EmptyInputService::~EmptyInputService() {
    delete retriever_;
  }

  std::auto_ptr<EventPrincipal>
  EmptyInputService::read() {
    std::auto_ptr<EventPrincipal> result(0);
    
    if ( remainingEvents_-- != 0 ) {
      result = std::auto_ptr<EventPrincipal>(new EventPrincipal(nextID_++, *retriever_));
    }
    return result;
  }
}
