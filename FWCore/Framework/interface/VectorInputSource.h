#ifndef Framework_VectorInputSource_h
#define Framework_VectorInputSource_h


/*----------------------------------------------------------------------
  
VectorInputSource: Abstract interface for vector input sources.
$Id: VectorInputSource.h,v 1.3 2006/01/19 22:27:14 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/GenericInputSource.h"

namespace edm {
  class EventPrincipal;
  class InputSourceDescription;
  class ParameterSet;
  class VectorInputSource : public GenericInputSource {
  public:
    typedef boost::shared_ptr<EventPrincipal> EventPrincipalVectorElement;
    typedef std::vector<EventPrincipalVectorElement> EventPrincipalVector;
    explicit VectorInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~VectorInputSource();

    void readMany(int number, EventPrincipalVector& result);

  private:
    virtual void readMany_(int number, EventPrincipalVector& result) = 0;
  };
}

#endif
