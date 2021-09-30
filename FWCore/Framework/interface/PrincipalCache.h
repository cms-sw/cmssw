#ifndef FWCore_Framework_PrincipalCache_h
#define FWCore_Framework_PrincipalCache_h

/*
Contains smart pointers to the RunPrincipals,
LuminosityBlockPrincipals, EventPrincipals,
and ProcessBlockPrincipals. It keeps the
objects alive so they can be reused as
necessary.

Original Author: W. David Dagenhart
*/

#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

#include <memory>
#include <vector>

namespace edm {

  class ProcessBlockPrincipal;
  class RunPrincipal;
  class LuminosityBlockPrincipal;
  class EventPrincipal;
  class ProductRegistry;
  class PreallocationConfiguration;

  class PrincipalCache {
  public:
    PrincipalCache();
    ~PrincipalCache();
    PrincipalCache(PrincipalCache&&) = default;

    ProcessBlockPrincipal& processBlockPrincipal() const { return *processBlockPrincipal_; }
    ProcessBlockPrincipal& inputProcessBlockPrincipal() const { return *inputProcessBlockPrincipal_; }

    enum class ProcessBlockType { New, Input };
    ProcessBlockPrincipal& processBlockPrincipal(ProcessBlockType processBlockType) const {
      return processBlockType == ProcessBlockType::Input ? *inputProcessBlockPrincipal_ : *processBlockPrincipal_;
    }

    std::shared_ptr<RunPrincipal> getAvailableRunPrincipalPtr();
    std::shared_ptr<LuminosityBlockPrincipal> getAvailableLumiPrincipalPtr();

    EventPrincipal& eventPrincipal(unsigned int iStreamIndex) const { return *(eventPrincipals_[iStreamIndex]); }

    void setNumberOfConcurrentPrincipals(PreallocationConfiguration const&);
    void insert(std::unique_ptr<ProcessBlockPrincipal>);
    void insertForInput(std::unique_ptr<ProcessBlockPrincipal>);
    void insert(std::unique_ptr<RunPrincipal>);
    void insert(std::unique_ptr<LuminosityBlockPrincipal>);
    void insert(std::shared_ptr<EventPrincipal>);

    void adjustEventsToNewProductRegistry(std::shared_ptr<ProductRegistry const>);

    void adjustIndexesAfterProductRegistryAddition();

  private:
    std::unique_ptr<ProcessBlockPrincipal> processBlockPrincipal_;
    std::unique_ptr<ProcessBlockPrincipal> inputProcessBlockPrincipal_;
    edm::ReusableObjectHolder<RunPrincipal> runHolder_;
    edm::ReusableObjectHolder<LuminosityBlockPrincipal> lumiHolder_;
    std::vector<std::shared_ptr<EventPrincipal>> eventPrincipals_;
  };
}  // namespace edm

#endif
