#ifndef FWCore_Framework_GetterOfProducts_h
#define FWCore_Framework_GetterOfProducts_h

/** \class edm::GetterOfProducts

Intended to be used by EDProducers, EDFilters, and
EDAnalyzers to get products from the Event, Run, LuminosityBlock
or ProcessBlock. In most cases, the preferred
method to get products is not to use this class. In
most cases the preferred method is to use the function
getByToken with a token obtained from a consumes call
which was passed a configurable InputTag. But
occasionally getByToken will not work because one
wants to select the product based on the data that is
available and not have to modify the
configuration as the data content changes. A real
example would be a module that collects HLT trigger
information from products written by the many HLT
filters. The number and labels of those products vary
so much that it would not be reasonable to modify
the configuration to get all of them each time a
different HLT trigger table was used. This class
handles that and similar cases.

This method can select by type and branch type.
There exists a predicate (in ProcessMatch.h)
to also select on process name.  It is possible
to write other predicates which will select on
anything in the BranchDescription. The selection
is done during the initialization of the process.
During this initialization a list of tokens
is filled with all matching products from the
ProductRegistry. This list of tokens is accessible
to the module.

The fillHandles functions will get a handle
for each product on the list of tokens that
is actually present in the current Event,
LuminosityBlock, Run, or ProcessBlock. Internally,
this function uses tokens and depends on the same
things as getByToken and benefits from
performance optimizations of getByToken.

Typically one would use this as follows:

Add these headers:

#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/ProcessMatch.h"

Add this data member:

    edm::GetterOfProducts<YourDataType> getterOfProducts_;


Add these to the constructor (1st line is usually in the
data member initializer list and the 2nd line in the body
of the constructor)

    getterOfProducts_(edm::ProcessMatch(processName_), this) {
    callWhenNewProductsRegistered(getterOfProducts_);

Add this to the method called for each event:

    std::vector<edm::Handle<YourDataType> > handles;
    getterOfProducts_.fillHandles(event, handles);

And that is all you need in most cases. In the above example,
"YourDataType" is the type of the product you want to get.
There are some variants for special cases

  - Use an extra argument to the constructor for products
  in a Run, LuminosityBlock or ProcessBlock For example:

    getterOfProducts_ = edm::GetterOfProducts<Thing>(edm::ProcessMatch(processName_), this, edm::InRun);

  - You can use multiple GetterOfProducts's in the same module. The
  only tricky part is to use a lambda as follows to register the
  callbacks:

    callWhenNewProductsRegistered([this](edm::BranchDescription const& bd) {
      getterOfProducts1_(bd);
      getterOfProducts2_(bd);
    });

  - One can use "*" for the processName_ to select from all
  processes (this will just select based on type).

  - You can define your own predicate to replace ProcessMatch
  in the above example and select based on anything in the
  BranchDescription. See ProcessMatch.h for an example of how
  to write this predicate.

\author W. David Dagenhart, created 6 August, 2012

*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/ProcessBlockForOutput.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/WillGetIfMatch.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace edm {

  template <typename U>
  struct BranchTypeForContainerType {
    static constexpr BranchType branchType = InEvent;
  };
  template <>
  struct BranchTypeForContainerType<LuminosityBlock> {
    static constexpr BranchType branchType = InLumi;
  };
  template <>
  struct BranchTypeForContainerType<LuminosityBlockForOutput> {
    static constexpr BranchType branchType = InLumi;
  };
  template <>
  struct BranchTypeForContainerType<Run> {
    static constexpr BranchType branchType = InRun;
  };
  template <>
  struct BranchTypeForContainerType<RunForOutput> {
    static constexpr BranchType branchType = InRun;
  };
  template <>
  struct BranchTypeForContainerType<ProcessBlock> {
    static constexpr BranchType branchType = InProcess;
  };
  template <>
  struct BranchTypeForContainerType<ProcessBlockForOutput> {
    static constexpr BranchType branchType = InProcess;
  };

  template <typename T>
  class GetterOfProducts {
  public:
    GetterOfProducts() : branchType_(edm::InEvent) {}

    template <typename U, typename M>
    GetterOfProducts(U const& match, M* module, edm::BranchType branchType = edm::InEvent)
        : matcher_(WillGetIfMatch<T>(match, module)),
          tokens_(new std::vector<edm::EDGetTokenT<T>>),
          branchType_(branchType) {}

    void operator()(edm::BranchDescription const& branchDescription) {
      if (branchDescription.dropped())
        return;
      if (branchDescription.branchType() == branchType_ &&
          branchDescription.unwrappedTypeID() == edm::TypeID(typeid(T))) {
        auto const& token = matcher_(branchDescription);
        if (not token.isUninitialized()) {
          tokens_->push_back(token);
        }
      }
    }

    template <typename ProductContainer>
    void fillHandles(ProductContainer const& productContainer, std::vector<edm::Handle<T>>& handles) const {
      handles.clear();
      if (branchType_ == BranchTypeForContainerType<ProductContainer>::branchType) {
        handles.reserve(tokens_->size());
        for (auto const& token : *tokens_) {
          if (auto handle = productContainer.getHandle(token)) {
            handles.push_back(handle);
          }
        }
      }
    }

    std::vector<edm::EDGetTokenT<T>> const& tokens() const { return *tokens_; }
    edm::BranchType branchType() const { return branchType_; }

  private:
    std::function<EDGetTokenT<T>(BranchDescription const&)> matcher_;
    // A shared pointer is needed because objects of this type get assigned
    // to std::function's and we want the copies in those to share the same vector.
    std::shared_ptr<std::vector<edm::EDGetTokenT<T>>> tokens_;
    edm::BranchType branchType_;
  };
}  // namespace edm
#endif
