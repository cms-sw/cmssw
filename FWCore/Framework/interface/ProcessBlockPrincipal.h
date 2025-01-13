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
    template <typename FACTORY>
      requires(requires(FACTORY&& f, std::string const& name, ProductRegistry const& reg) { f(InProcess, name, reg); })
    ProcessBlockPrincipal(std::shared_ptr<ProductRegistry const> iReg,
                          FACTORY&& iFactory,
                          ProcessConfiguration const& iConfig)
        : ProcessBlockPrincipal(iReg, iFactory(InProcess, iConfig.processName(), *iReg), iConfig) {}

    void fillProcessBlockPrincipal(std::string const& processName, DelayedReader* reader = nullptr);

    std::string const& processName() const { return processName_; }

    void put(ProductResolverIndex index, std::unique_ptr<WrapperBase> edp) const;

    // Should only be 1 ProcessBlock needed at a time (no concurrent ProcessBlocks)
    unsigned int index() const { return 0; }

  private:
    ProcessBlockPrincipal(std::shared_ptr<ProductRegistry const>,
                          std::vector<std::shared_ptr<ProductResolverBase>>&& resolvers,
                          ProcessConfiguration const&);
    unsigned int transitionIndex_() const final;

    std::string processName_;
  };
}  // namespace edm

#endif  // FWCore_Framework_ProcessBlockPrincipal_h
