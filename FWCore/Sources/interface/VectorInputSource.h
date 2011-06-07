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

    template<typename T>
    size_t loopRandom(size_t number, T eventOperator);
    template<typename T>
    size_t loopSequential(size_t number, T eventOperator);
    template<typename T, typename Collection>
    size_t loopSpecified(Collection const& events, T eventOperator);

    void readMany(int number, EventPrincipalVector& result);
    void readManyRandom(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    void readManySequential(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber);
    void readManySpecified(std::vector<EventID> const& events, EventPrincipalVector& result);
    void dropUnwantedBranches(std::vector<std::string> const& wantedBranches);

  private:

    virtual EventPrincipal* readOneRandom() = 0;
    virtual EventPrincipal* readOneSequential() = 0;
    virtual EventPrincipal* readOneSpecified(EventID const& event) = 0;

    virtual void readMany_(int number, EventPrincipalVector& result) = 0;
    virtual void readManyRandom_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) = 0;
    virtual void readManySequential_(int number, EventPrincipalVector& result, unsigned int& fileSeqNumber) = 0;
    virtual void readManySpecified_(std::vector<EventID> const& events, EventPrincipalVector& result) = 0;
    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) = 0;
  };

  template<typename T>
  size_t VectorInputSource::loopRandom(size_t number, T eventOperator) {
    size_t i = 0U;
    for(; i < number; ++i) {
      EventPrincipal* ep = readOneRandom();
      if(!ep) break;
      eventOperator(*ep);
    }
    return i;
  }

  template<typename T>
  size_t VectorInputSource::loopSequential(size_t number, T eventOperator) {
    size_t i = 0U;
    for(; i < number; ++i) {
      EventPrincipal* ep = readOneSequential();
      if(!ep) break;
      eventOperator(*ep);
    }
    return i;
  }

  template<typename T, typename Collection>
  size_t VectorInputSource::loopSpecified(Collection const& events, T eventOperator) {
    size_t i = 0U;
    for(typename Collection::const_iterator it = events.begin(), itEnd = events.end(); it != itEnd; ++it) {
      EventPrincipal* ep = readOneSpecified(*it);
      if(!ep) break;
      eventOperator(*ep);
      ++i;
    }
    return i;
  }
}
#endif
