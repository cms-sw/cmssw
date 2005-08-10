/*----------------------------------------------------------------------
$Id: EmptyInputService.h,v 1.3 2005/07/14 21:34:44 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Retriever.h"
#include "FWCore/Framework/interface/InputService.h"
#include "FWCore/EDProduct/interface/EventID.h"

namespace edm {
  class ParameterSet;

  class EmptyInputService : public InputService {
  public:
    explicit EmptyInputService(ParameterSet const&, const InputServiceDescription&);
    ~EmptyInputService();
  private:
    std::auto_ptr<EventPrincipal> read();
    
    EventID nextID_;
    int remainingEvents_;
    Retriever* retriever_;
  };


  struct FakeRetriever : public Retriever {
    virtual ~FakeRetriever();
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k);
  };
}
