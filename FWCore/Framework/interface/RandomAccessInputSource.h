#ifndef Framework_RandomAccessInputSource_h
#define Framework_RandomAccessInputSource_h


/*----------------------------------------------------------------------
  
RandomAccessInputSource: Abstract interface for all random access input sources.
$Id: RandomAccessInputSource.h,v 1.6 2005/09/01 23:30:49 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/InputSource.h"

namespace edm {
  class EventID;
  class EventPrincipal;
  class InputSourceDescription;
  class RandomAccessInputSource : public InputSource {
  public:
    explicit RandomAccessInputSource(InputSourceDescription const&);
    virtual ~RandomAccessInputSource();

    std::auto_ptr<EventPrincipal> readEvent(EventID const& id);
    void skipEvents(int offset);

  private:
    virtual std::auto_ptr<EventPrincipal> read(EventID const&) = 0;

    virtual void skip(int) = 0;
  };
}

#endif
