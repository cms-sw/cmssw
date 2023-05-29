#ifndef FWCore_Framework_ProducerBase_h
#define FWCore_Framework_ProducerBase_h

/*----------------------------------------------------------------------

EDProducer: The base class of all "modules" that will insert new
EDProducts into an Event.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"

#include <functional>
#include <unordered_map>
#include <string>
#include <vector>
#include <array>

namespace edm {
  class BranchDescription;
  class ModuleDescription;
  class ProducesCollector;
  class ProductRegistry;
  class Event;
  class LuminosityBlock;
  class ProcessBlock;
  class Run;

  class EDProducer;
  namespace one {
    class EDProducerBase;
    class EDFilterBase;
  }  // namespace one
  namespace global {
    class EDProducerBase;
    class EDFilterBase;
  }  // namespace global
  namespace limited {
    class EDProducerBase;
    class EDFilterBase;
  }  // namespace limited
  namespace stream {
    template <typename T>
    class ProducingModuleAdaptorBase;
  }

  namespace producerbasehelper {
    template <typename P>
    struct PrincipalTraits;
    template <>
    struct PrincipalTraits<ProcessBlock> {
      static constexpr int kBranchType = InProcess;
    };
    template <>
    struct PrincipalTraits<Run> {
      static constexpr int kBranchType = InRun;
    };
    template <>
    struct PrincipalTraits<LuminosityBlock> {
      static constexpr int kBranchType = InLumi;
    };
    template <>
    struct PrincipalTraits<Event> {
      static constexpr int kBranchType = InEvent;
    };
  }  // namespace producerbasehelper

  class ProducerBase : private ProductRegistryHelper {
  public:
    typedef ProductRegistryHelper::TypeLabelList TypeLabelList;
    ProducerBase();
    ~ProducerBase() noexcept(false) override;

    /// used by the fwk to register list of products
    std::function<void(BranchDescription const&)> registrationCallback() const;

    void registerProducts(ProducerBase*, ProductRegistry*, ModuleDescription const&);

    using ProductRegistryHelper::recordProvenanceList;
    using ProductRegistryHelper::typeLabelList;

    template <typename T>
    using BranchAliasSetterT = ProductRegistryHelper::BranchAliasSetterT<T>;

    void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func) {
      callWhenNewProductsRegistered_ = func;
    }

    using ModuleToResolverIndicies =
        std::unordered_multimap<std::string, std::tuple<edm::TypeID const*, const char*, edm::ProductResolverIndex>>;
    void resolvePutIndicies(BranchType iBranchType,
                            ModuleToResolverIndicies const& iIndicies,
                            std::string const& moduleLabel);

    std::vector<edm::ProductResolverIndex> const& indiciesForPutProducts(BranchType iBranchType) const {
      return putIndicies_[iBranchType];
    }

    std::vector<edm::ProductResolverIndex> const& putTokenIndexToProductResolverIndex() const {
      return putTokenToResolverIndex_;
    }

  protected:
    ProducesCollector producesCollector();
    using ProductRegistryHelper::produces;

  private:
    friend class one::EDProducerBase;
    friend class one::EDFilterBase;
    friend class global::EDProducerBase;
    friend class global::EDFilterBase;
    friend class limited::EDProducerBase;
    friend class limited::EDFilterBase;
    friend class PuttableSourceBase;
    friend class TransformerBase;
    template <typename T>
    friend class stream::ProducingModuleAdaptorBase;

    template <typename P>
    void commit_(P& iPrincipal) {
      iPrincipal.commit_(putIndicies_[producerbasehelper::PrincipalTraits<P>::kBranchType]);
    }

    template <typename P, typename I>
    void commit_(P& iPrincipal, I* iID) {
      iPrincipal.commit_(putIndicies_[producerbasehelper::PrincipalTraits<P>::kBranchType], iID);
    }

    using ProductRegistryHelper::transforms;

    std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;
    std::array<std::vector<edm::ProductResolverIndex>, edm::NumBranchTypes> putIndicies_;
    std::vector<edm::ProductResolverIndex> putTokenToResolverIndex_;
  };
}  // namespace edm
#endif
