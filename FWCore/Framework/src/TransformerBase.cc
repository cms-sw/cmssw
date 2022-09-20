#include "FWCore/Framework/interface/TransformerBase.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/EventForTransformer.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace edm {
  void TransformerBase::registerTransformImp(
      ProducerBase& iBase, EDPutToken iToken, const TypeID& id, std::string instanceName, TransformFunction iFunc) {
    auto transformPut = iBase.transforms(id, std::move(instanceName));
    transformInfo_.emplace_back(iToken.index(), id, transformPut, std::move(iFunc));
  }

  std::size_t TransformerBase::findMatchingIndex(ProducerBase const& iBase,
                                                 edm::BranchDescription const& iBranch) const {
    auto const& list = iBase.typeLabelList();

    std::size_t index = 0;
    bool found = false;
    for (auto const& element : list) {
      if (not element.isTransform_) {
        continue;
      }
      if (element.typeID_ == iBranch.unwrappedTypeID() &&
          element.productInstanceName_ == iBranch.productInstanceName()) {
        found = true;
        break;
      }
      ++index;
    }
    assert(found);
    return index;
  }

  void TransformerBase::extendUpdateLookup(ProducerBase const& iBase,
                                           ModuleDescription const& iModuleDesc,
                                           ProductResolverIndexHelper const& iHelper) {
    auto const& list = iBase.typeLabelList();

    for (auto it = transformInfo_.begin<0>(); it != transformInfo_.end<0>(); ++it) {
      auto const& putInfo = list[*it];
      *it = iHelper.index(PRODUCT_TYPE,
                          putInfo.typeID_,
                          iModuleDesc.moduleLabel().c_str(),
                          putInfo.productInstanceName_.c_str(),
                          iModuleDesc.processName().c_str());
    }
  }

  void TransformerBase::transformImp(std::size_t iIndex,
                                     ProducerBase const& iBase,
                                     edm::EventForTransformer& iEvent) const {
    auto handle = iEvent.get(transformInfo_.get<1>(iIndex), transformInfo_.get<0>(iIndex));

    if (handle.wrapper()) {
      iEvent.put(iBase.putTokenIndexToProductResolverIndex()[transformInfo_.get<2>(iIndex).index()],
                 transformInfo_.get<3>(iIndex)(*handle.wrapper()),
                 handle);
    }
  }

}  // namespace edm
