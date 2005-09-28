#ifndef Framework_EmptySource_h
#define Framework_EmptySource_h

/*----------------------------------------------------------------------
$Id: EmptySource.h,v 1.1 2005/09/07 19:09:26 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/RandomAccessInputSource.h"
#include "FWCore/EDProduct/interface/EventID.h"

namespace edm {
  class ParameterSet;

  class EmptySource : public RandomAccessInputSource {
  public:
    explicit EmptySource(ParameterSet const&, const InputSourceDescription&);
    ~EmptySource();
  private:
    std::auto_ptr<EventPrincipal> read();

    std::auto_ptr<EventPrincipal> read(EventID const&);

    void skip(int offset);
    
    int remainingEvents_;
    
    unsigned long numberEventsInRun_;
    unsigned long presentRun_;
    unsigned long long nextTime_;
    unsigned long timeBetweenEvents_;

    unsigned long numberEventsInThisRun_;
    EventID nextID_;
  };
}
#endif
