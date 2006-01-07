#ifndef Framework_VectorInputSource_h
#define Framework_VectorInputSource_h


/*----------------------------------------------------------------------
  
VectorInputSource: Abstract interface for secondary input sources.
$Id: VectorInputSource.h,v 1.3 2006/01/06 00:30:38 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/InputSource.h"

namespace edm {
  class EventPrincipal;
  class InputSourceDescription;
  class VectorInputSource : public InputSource {
  public:
    explicit VectorInputSource(InputSourceDescription const& desc);
    virtual ~VectorInputSource();

    void readMany(int number, std::vector<EventPrincipal*>& result);

  private:
    virtual void readMany_(int number, std::vector<EventPrincipal*>& result) = 0;
  };
}

#endif
