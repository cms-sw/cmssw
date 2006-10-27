#ifndef Framework_VectorInputSource_h
#define Framework_VectorInputSource_h


/*----------------------------------------------------------------------
  
VectorInputSource: Abstract interface for vector input sources.
$Id: VectorInputSource.h,v 1.5 2006/04/06 23:26:28 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/EDInputSource.h"
#include "FWCore/Framework/interface/EventPrincipalFwd.h"

namespace edm {
  class InputSourceDescription;
  class ParameterSet;
  class VectorInputSource : public EDInputSource {
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
