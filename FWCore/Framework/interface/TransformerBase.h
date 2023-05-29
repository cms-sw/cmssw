//
//  TransformerBase.h
//  CMSSW
//
//  Created by Chris Jones on 6/02/22.
//

#ifndef FWCore_Framework_TransformerBase_h
#define FWCore_Framework_TransformerBase_h

#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/SoATuple.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/ProductResolverIndex.h"

#include <string>
#include <functional>
#include <memory>
#include <any>

namespace edm {
  class ProducerBase;
  class TypeID;
  class WrapperBase;
  class EventForTransformer;
  class BranchDescription;
  class ProductResolverIndexHelper;
  class ModuleDescription;
  class WaitingTaskWithArenaHolder;
  class WaitingTaskHolder;

  class TransformerBase {
  public:
    TransformerBase() = default;
    virtual ~TransformerBase() noexcept(false) = default;

  protected:
    //The function takes the WrapperBase corresponding to the data product from the EDPutToken
    // and returns the WrapperBase associated to the id and instanceName
    using TransformFunction = std::function<std::unique_ptr<edm::WrapperBase>(std::any)>;
    using PreTransformFunction = std::function<std::any(edm::WrapperBase const&, edm::WaitingTaskWithArenaHolder)>;

    void registerTransformImp(ProducerBase&, EDPutToken, const TypeID& id, std::string instanceName, TransformFunction);
    void registerTransformAsyncImp(
        ProducerBase&, EDPutToken, const TypeID& id, std::string instanceName, PreTransformFunction, TransformFunction);

    std::size_t findMatchingIndex(ProducerBase const& iBase, edm::BranchDescription const&) const;
    ProductResolverIndex prefetchImp(std::size_t iIndex) const { return transformInfo_.get<kResolverIndex>(iIndex); }
    void transformImpAsync(WaitingTaskHolder iTask,
                           std::size_t iIndex,
                           ProducerBase const& iBase,
                           edm::EventForTransformer&) const;

    void extendUpdateLookup(ProducerBase const&,
                            ModuleDescription const& iModuleDesc,
                            ProductResolverIndexHelper const& iHelper);

  private:
    enum InfoColumns { kResolverIndex, kType, kToken, kPreTransform, kTransform };
    SoATuple<ProductResolverIndex, TypeID, EDPutToken, PreTransformFunction, TransformFunction> transformInfo_;
  };
}  // namespace edm

#endif /* TransformerBase_h */
