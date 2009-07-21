#ifndef FWCore_Sources_VectorInputSource_h
#define FWCore_Sources_VectorInputSource_h


/*----------------------------------------------------------------------
  
VectorInputSource: Abstract interface for vector input sources.
$Id: VectorInputSource.h,v 1.5 2008/05/29 16:17:48 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>
#include <vector>
#include "boost/shared_ptr.hpp"

#include "FWCore/Sources/interface/EDInputSource.h"

namespace edm {
  class EventPrincipal;
  class InputSourceDescription;
  class ParameterSet;
  class VectorInputSource : public EDInputSource {
  public:
    typedef boost::shared_ptr<EventPrincipal> EventPrincipalVectorElement;
    typedef std::vector<EventPrincipalVectorElement> EventPrincipalVector;
    explicit VectorInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~VectorInputSource();

    void readMany(int number, EventPrincipalVector& result);
    void readMany(int number, EventPrincipalVector& result, EventID const& id, unsigned int fileSeqNumber);
    void readManyRandom(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber); 
    void readManySequential(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber); 
    void dropUnwantedBranches(std::vector<std::string> const& wantedBranches);

  private:
    virtual void readMany_(int number, EventPrincipalVector& result) = 0;
    virtual void readMany_(int number, EventPrincipalVector& result, EventID const& id, unsigned int fileSeqNumber) = 0;
    virtual void readManyRandom_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) = 0;
    virtual void readManySequential_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) = 0;
    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) = 0;
  };
}

#endif
