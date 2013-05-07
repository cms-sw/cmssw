#ifndef FWCore_Sources_VectorInputSource_h
#define FWCore_Sources_VectorInputSource_h

/*----------------------------------------------------------------------
VectorInputSource: Abstract interface for vector input sources.
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/EDInputSource.h"

#include <memory>
#include <string>
#include <vector>

namespace edm {
  class EventPrincipal;
  struct InputSourceDescription;
  class LuminosityBlockID;
  class ParameterSet;
  class VectorInputSource : public EDInputSource {
  public:
    explicit VectorInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~VectorInputSource();

    template<typename T>
    size_t loopRandom(EventPrincipal& cache, size_t number, T eventOperator);
    template<typename T>
    size_t loopSequential(EventPrincipal& cache, size_t number, T eventOperator);
    template<typename T>
    size_t loopRandomWithID(EventPrincipal& cache, LuminosityBlockID const& id, size_t number, T eventOperator);
    template<typename T>
    size_t loopSequentialWithID(EventPrincipal& cache, LuminosityBlockID const& id, size_t number, T eventOperator);
    template<typename T, typename Collection>
    size_t loopSpecified(EventPrincipal& cache, Collection const& events, T eventOperator);

    void dropUnwantedBranches(std::vector<std::string> const& wantedBranches);

  private:

    void clearEventPrincipal(EventPrincipal& cache);
    virtual EventPrincipal* readOneRandom(EventPrincipal& cache) = 0;
    virtual EventPrincipal* readOneRandomWithID(EventPrincipal& cache, LuminosityBlockID const& id) = 0;
    virtual EventPrincipal* readOneSequential(EventPrincipal& cache) = 0;
    virtual EventPrincipal* readOneSequentialWithID(EventPrincipal& cache, LuminosityBlockID const& id) = 0;
    virtual EventPrincipal* readOneSpecified(EventPrincipal& cache, EventID const& event) = 0;

    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) = 0;
  };

  template<typename T>
  size_t VectorInputSource::loopRandom(EventPrincipal& cache, size_t number, T eventOperator) {
    size_t i = 0U;
    for(; i < number; ++i) {
      EventPrincipal* ep = readOneRandom(cache);
      if(!ep) break;
      eventOperator(*ep);
      clearEventPrincipal(cache);
    }
    return i;
  }

  template<typename T>
  size_t VectorInputSource::loopSequential(EventPrincipal& cache, size_t number, T eventOperator) {
    size_t i = 0U;
    for(; i < number; ++i) {
      EventPrincipal* ep = readOneSequential(cache);
      if(!ep) break;
      eventOperator(*ep);
      clearEventPrincipal(cache);
    }
    return i;
  }

  template<typename T>
  size_t VectorInputSource::loopRandomWithID(EventPrincipal& cache, LuminosityBlockID const& id, size_t number, T eventOperator) {
    size_t i = 0U;
    for(; i < number; ++i) {
      EventPrincipal* ep = readOneRandomWithID(cache, id);
      if(!ep) break;
      eventOperator(*ep);
      clearEventPrincipal(cache);
    }
    return i;
  }

  template<typename T>
  size_t VectorInputSource::loopSequentialWithID(EventPrincipal& cache, LuminosityBlockID const& id, size_t number, T eventOperator) {
    size_t i = 0U;
    for(; i < number; ++i) {
      EventPrincipal* ep = readOneSequentialWithID(cache, id);
      if(!ep) break;
      eventOperator(*ep);
      clearEventPrincipal(cache);
    }
    return i;
  }

  template<typename T, typename Collection>
  size_t VectorInputSource::loopSpecified(EventPrincipal& cache, Collection const& events, T eventOperator) {
    size_t i = 0U;
    for(typename Collection::const_iterator it = events.begin(), itEnd = events.end(); it != itEnd; ++it) {
      EventPrincipal* ep = readOneSpecified(cache, *it);
      if(!ep) break;
      eventOperator(*ep);
      ++i;
      clearEventPrincipal(cache);
    }
    return i;
  }
}
#endif
