#ifndef Framework_VectorInputSource_h
#define Framework_VectorInputSource_h


/*----------------------------------------------------------------------
  
VectorInputSource: Abstract interface for secondary input sources.
$Id: VectorInputSource.h,v 1.1 2006/01/07 20:41:11 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/InputSource.h"

namespace edm {
  class EventPrincipal;
  class InputSourceDescription;
  class VectorInputSource : public InputSource {
  public:
    typedef boost::shared_ptr<EventPrincipal> EventPrincipalVectorElement;
    typedef std::vector<EventPrincipalVectorElement> EventPrincipalVector;
    explicit VectorInputSource(InputSourceDescription const& desc);
    virtual ~VectorInputSource();

    void readMany(int number, EventPrincipalVector& result);

  private:
    virtual void readMany_(int number, EventPrincipalVector& result) = 0;
  };
}

#endif
