#ifndef FWCore_Sources_VectorInputSource_h
#define FWCore_Sources_VectorInputSource_h

/*----------------------------------------------------------------------
VectorInputSource: Abstract interface for vector input sources.
----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/SecondaryEventIDAndFileInfo.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <string>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class EventPrincipal;
  struct VectorInputSourceDescription;
  class EventID;
  class ParameterSet;
  class VectorInputSource {
  public:
    explicit VectorInputSource(ParameterSet const& pset, VectorInputSourceDescription const& desc);
    virtual ~VectorInputSource();

    template<typename T>
    size_t loopOverEvents(EventPrincipal& cache, size_t& fileNameHash, size_t number, T eventOperator, CLHEP::HepRandomEngine* = nullptr, EventID const* id = nullptr);

    template<typename T, typename Iterator>
    size_t loopSpecified(EventPrincipal& cache, size_t& fileNameHash, Iterator const& begin, Iterator const& end, T eventOperator);

    void dropUnwantedBranches(std::vector<std::string> const& wantedBranches);
    //
    /// Called at beginning of job
    void doBeginJob();

    /// Called at end of job
    void doEndJob();

    std::shared_ptr<ProductRegistry const> productRegistry() const {return productRegistry_;}
    ProductRegistry& productRegistryUpdate() const {return *productRegistry_;}
    ProcessHistoryRegistry const& processHistoryRegistry() const {return *processHistoryRegistry_;}
    ProcessHistoryRegistry& processHistoryRegistryForUpdate() {return *processHistoryRegistry_;}

  private:

    void clearEventPrincipal(EventPrincipal& cache);

  private:
    virtual bool readOneEvent(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*, EventID const* id) = 0;
    virtual void readOneSpecified(EventPrincipal& cache, size_t& fileNameHash, SecondaryEventIDAndFileInfo const& event) = 0;
    void readOneSpecified(EventPrincipal& cache, size_t& fileNameHash, EventID const& event) {
      SecondaryEventIDAndFileInfo info(event, fileNameHash);
      readOneSpecified(cache, fileNameHash, info);
    }

    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) = 0;
    virtual void beginJob() = 0;
    virtual void endJob() = 0;

    std::shared_ptr<ProductRegistry> productRegistry_;
    std::unique_ptr<ProcessHistoryRegistry> processHistoryRegistry_;
  };

  template<typename T>
  size_t VectorInputSource::loopOverEvents(EventPrincipal& cache, size_t& fileNameHash, size_t number, T eventOperator, CLHEP::HepRandomEngine* engine, EventID const* id) {
    size_t i = 0U;
    for(; i < number; ++i) {
      clearEventPrincipal(cache);
      bool found = readOneEvent(cache, fileNameHash, engine, id);
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
