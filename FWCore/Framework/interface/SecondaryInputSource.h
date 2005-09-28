#ifndef Framework_SecondaryInputSource_h
#define Framework_SecondaryInputSource_h


/*----------------------------------------------------------------------
  
SecondaryInputSource: Abstract interface for secondary input sources.
$Id: SecondaryInputSource.h,v 1.6 2005/09/01 23:30:49 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include "boost/shared_ptr.hpp"
#include "FWCore/Framework/interface/InputSource.h"

// Note: The inheritance from inputSource is needed to make the factory work.
// We should remove this inheritance, and give the secondary source its own factory.

namespace edm {
  class EventPrincipal;
  class SecondaryInputSource : public InputSource {
  public:
    SecondaryInputSource();
    explicit SecondaryInputSource(InputSourceDescription const&); // KLUDGE
    virtual ~SecondaryInputSource();

    void readMany(int idx, int number, std::vector<EventPrincipal*>& result);

  private:
    virtual std::auto_ptr<EventPrincipal> read(); //KLUDGE
    virtual void read(int idx, int number, std::vector<EventPrincipal*>& result) = 0;
  };
}

#endif
