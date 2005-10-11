#ifndef Framework_SecondaryInputSource_h
#define Framework_SecondaryInputSource_h


/*----------------------------------------------------------------------
  
SecondaryInputSource: Abstract interface for secondary input sources.
$Id: SecondaryInputSource.h,v 1.1 2005/09/28 05:17:39 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include "boost/shared_ptr.hpp"

namespace edm {
  class EventPrincipal;
  class SecondaryInputSource {
  public:
    SecondaryInputSource();
    virtual ~SecondaryInputSource();

    void readMany(int idx, int number, std::vector<EventPrincipal*>& result);

  private:
    virtual void read(int idx, int number, std::vector<EventPrincipal*>& result) = 0;
  };
}

#endif
