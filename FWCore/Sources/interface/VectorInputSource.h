#ifndef FWCore_Sources_VectorInputSource_h
#define FWCore_Sources_VectorInputSource_h

/*----------------------------------------------------------------------
VectorInputSource: Abstract interface for vector input sources.
----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/SecondaryEventIDAndFileInfo.h"
#include "FWCore/Sources/interface/EDInputSource.h"

#include <memory>
#include <string>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

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
    size_t loopRandom(EventPrincipal& cache, size_t& fileNameHash, size_t number, T eventOperator, CLHEP::HepRandomEngine*);
    template<typename T>
    size_t loopSequential(EventPrincipal& cache, size_t& fileNameHash, size_t number, T eventOperator);
    template<typename T>
    size_t loopRandomWithID(EventPrincipal& cache, size_t& fileNameHash, LuminosityBlockID const& id, size_t number, T eventOperator, CLHEP::HepRandomEngine*);
    template<typename T>
    size_t loopSequentialWithID(EventPrincipal& cache, size_t& fileNameHash, LuminosityBlockID const& id, size_t number, T eventOperator);
    template<typename T, typename Iterator>
    size_t loopSpecified(EventPrincipal& cache, size_t& fileNameHash, Iterator const& begin, Iterator const& end, T eventOperator);

    void dropUnwantedBranches(std::vector<std::string> const& wantedBranches);

  private:

    void clearEventPrincipal(EventPrincipal& cache);
    virtual void readOneRandom(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*) = 0;
    virtual bool readOneRandomWithID(EventPrincipal& cache, size_t& fileNameHash, LuminosityBlockID const& id, CLHEP::HepRandomEngine*) = 0;
    virtual bool readOneSequential(EventPrincipal& cache, size_t& fileNameHash) = 0;
    virtual bool readOneSequentialWithID(EventPrincipal& cache, size_t& fileNameHash, LuminosityBlockID const& id) = 0;
    virtual void readOneSpecified(EventPrincipal& cache, size_t& fileNameHash, SecondaryEventIDAndFileInfo const& event) = 0;
    void readOneSpecified(EventPrincipal& cache, size_t& fileNameHash, EventID const& event) {
      SecondaryEventIDAndFileInfo info(event, fileNameHash);
      readOneSpecified(cache, fileNameHash, info);
    }

    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) = 0;
  };

  template<typename T>
  size_t VectorInputSource::loopRandom(EventPrincipal& cache, size_t& fileNameHash, size_t number, T eventOperator, CLHEP::HepRandomEngine* engine) {
    size_t i = 0U;
    for(; i < number; ++i) {
      clearEventPrincipal(cache);
      readOneRandom(cache, fileNameHash, engine);
      eventOperator(cache, fileNameHash);
    }
    return i;
  }

  template<typename T>
  size_t VectorInputSource::loopSequential(EventPrincipal& cache, size_t& fileNameHash, size_t number, T eventOperator) {
    size_t i = 0U;
    for(; i < number; ++i) {
      clearEventPrincipal(cache);
      bool found = readOneSequential(cache, fileNameHash);
      if(!found) break;
      eventOperator(cache, fileNameHash);
    }
    return i;
  }

  template<typename T>
  size_t VectorInputSource::loopRandomWithID(EventPrincipal& cache, size_t& fileNameHash, LuminosityBlockID const& id, size_t number, T eventOperator, CLHEP::HepRandomEngine* engine) {
    size_t i = 0U;
    for(; i < number; ++i) {
      clearEventPrincipal(cache);
      bool found = readOneRandomWithID(cache, fileNameHash, id, engine);
      if(!found) break;
      eventOperator(cache, fileNameHash);
    }
    return i;
  }

  template<typename T>
  size_t VectorInputSource::loopSequentialWithID(EventPrincipal& cache, size_t& fileNameHash, LuminosityBlockID const& id, size_t number, T eventOperator) {
    size_t i = 0U;
    for(; i < number; ++i) {
      clearEventPrincipal(cache);
      bool found = readOneSequentialWithID(cache, fileNameHash, id);
      if(!found) break;
      eventOperator(cache, fileNameHash);
    }
    return i;
  }

  template<typename T, typename Iterator>
  size_t VectorInputSource::loopSpecified(EventPrincipal& cache, size_t& fileNameHash, Iterator const& begin, Iterator const& end, T eventOperator) {
    size_t i = 0U;
    for(Iterator iter = begin; iter != end; ++iter) {
      clearEventPrincipal(cache);
      readOneSpecified(cache, fileNameHash, *iter);
      eventOperator(cache, fileNameHash);
      ++i;
    }
    return i;
  }
}
#endif
