#ifndef FWCore_Framework_ProcessBlockPrincipal_h
#define FWCore_Framework_ProcessBlockPrincipal_h

/** \class edm::ProcessBlockPrincipal

\author W. David Dagenhart, created 19 March, 2020

*/

#include "FWCore/Framework/interface/Principal.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"

#include <memory>
#include <string>

namespace edm {

  class ProcessConfiguration;
  class ProductRegistry;
  class WrapperBase;

  class ProcessBlockPrincipal : public Principal {
  public:
    ProcessBlockPrincipal(std::shared_ptr<ProductRegistry const>,
                          ProcessConfiguration const&,
                          bool isForPrimaryProcess = true);

    void fillProcessBlockPrincipal(std::string const& processName, DelayedReader* reader = nullptr);

    std::string const& processName() const { return processName_; }

    void put(ProductResolverIndex index, std::unique_ptr<WrapperBase> edp) const;

    // Should only be 1 ProcessBlock needed at a time (no concurrent ProcessBlocks)
    unsigned int index() const { return 0; }

  private:
    unsigned int transitionIndex_() const final;

    std::string processName_;
  };
}  // namespace edm

#endif  // FWCore_Framework_ProcessBlockPrincipal_h
