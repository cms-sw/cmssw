#ifndef Framework_EmptyInputService_h
#define Framework_EmptyInputService_h

/*----------------------------------------------------------------------
$Id: EmptyInputService.h,v 1.5 2005/08/10 15:28:12 chrjones Exp $
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
    
    int remainingEvents_;
    Retriever* retriever_;
    
    unsigned long numberEventsInRun_;
    unsigned long presentRun_;
    unsigned long long nextTime_;
    unsigned long timeBetweenEvents_;

    unsigned long numberEventsInThisRun_;
    EventID nextID_;

  };


  struct FakeRetriever : public Retriever {
    virtual ~FakeRetriever();
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k);
  };
}
#endif
