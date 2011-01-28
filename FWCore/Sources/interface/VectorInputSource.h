#ifndef FWCore_Sources_VectorInputSource_h
#define FWCore_Sources_VectorInputSource_h

/*----------------------------------------------------------------------
VectorInputSource: Abstract interface for vector input sources.
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/EDInputSource.h"

#include "boost/shared_ptr.hpp"

#include <memory>
#include <string>
#include <vector>

namespace edm {
  class EventPrincipal;
  struct InputSourceDescription;
  class ParameterSet;
  class VectorInputSource : public EDInputSource {
  public:
    typedef boost::shared_ptr<EventPrincipal> EventPrincipalVectorElement;
    typedef std::vector<EventPrincipalVectorElement> EventPrincipalVector;
    explicit VectorInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~VectorInputSource();

    void readMany(int number, EventPrincipalVector& result);
    void readManyRandom(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    void readManySequential(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    void readManySpecified(std::vector<EventID> const& events, EventPrincipalVector& result);
    void dropUnwantedBranches(std::vector<std::string> const& wantedBranches);

  private:
    virtual void readMany_(int number, EventPrincipalVector& result) = 0;
    virtual void readManyRandom_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) = 0;
    virtual void readManySequential_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) = 0;
    virtual void readManySpecified_(std::vector<EventID> const& events, EventPrincipalVector& result) = 0;
    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) = 0;
  };
}
#endif
