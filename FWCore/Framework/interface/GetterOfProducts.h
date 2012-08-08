#ifndef FWCore_Framework_GetterOfProducts_h
#define FWCore_Framework_GetterOfProducts_h

/** \class edm::GetterOfProducts

Intended to be used by EDProducers, EDFilters, and
EDAnalyzers to get products from the Event, Run
or LuminosityBlock. In most cases, the preferred
method to get products is not to use this class. In
most cases the preferred method is to use the function
getByLabel with an InputTag that is configurable. But
occasionally getByLabel will not work because one
wants to select the product based on the data that is
available in the event and not have to modify the
configuration as the data content changes. A real
example would be a module that collects HLT trigger
information from products written by the many HLT
filters. The number and labels of those products vary
so much that it would not be reasonable to modify
the configuration to get all of them each time a
different HLT trigger table was used. This class
handles that and similar cases.

This class is preferred over using getByType,
getManyByType, getBySelector, and getMany.
Those methods are deprecated and may be deleted
if we ever complete the migration remove all
uses of them.

This method can select by type and branch type.
There exists a predicate (in ProcessMatch.h)
to also select on process name.  It is possible
to write other predicates which will select on
anything in the BranchDescription. The selection
is done during the initialization of the process.
During this initialization a list of InputTags
is filled with all matching products from the
ProductRegistry. This list of InputTags is accessible
to the module. In the future there are plans
for modules to register the products they might
get to the Framework which will allow it to
optimize performance for parallel processing
among other things.

The fillHandles functions will get a handle
for each product on the list of InputTags that
is actually present in the current Event,
LuminosityBlock, or Run. Internally, this
function uses getByLabel and benefits from
performance optimizations of getByLabel.

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
  in the Run or LuminosityBlock.

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
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/WillGetIfMatch.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace edm {

   template <typename T>
   class GetterOfProducts {
   public:

      GetterOfProducts() : branchType_(edm::InEvent) { }

      template <typename U, typename M>
      GetterOfProducts(U const& match, M* module, edm::BranchType branchType = edm::InEvent) : 
        matcher_(WillGetIfMatch<T, M>(match, module)),
        inputTags_(new std::vector<edm::InputTag>),
        branchType_(branchType) {
      }

      void operator()(edm::BranchDescription const& branchDescription) {

         if (branchDescription.typeID() == edm::TypeID(typeid(T)) &&
             branchDescription.branchType() == branchType_ &&
             matcher_(branchDescription)) {

            inputTags_->emplace_back(branchDescription.moduleLabel(),
                                     branchDescription.productInstanceName(),
                                     branchDescription.processName());
         }
      }

      void fillHandles(edm::Event const& event, std::vector<edm::Handle<T> >& handles) const {
         handles.clear();
         handles.reserve(inputTags_->size());
         edm::Handle<T> handle;
         for (auto const& inputTag : *inputTags_) {
            event.getByLabel(inputTag, handle);
            if (handle.isValid()) {
               handles.push_back(handle);
            }
         }
      }

      void fillHandles(edm::LuminosityBlock const& lumi, std::vector<edm::Handle<T> >& handles) const {
         handles.clear();
         handles.reserve(inputTags_->size());
         edm::Handle<T> handle;
         for (auto const& inputTag : *inputTags_) {
            lumi.getByLabel(inputTag, handle);
            if (handle.isValid()) {
               handles.push_back(handle);
            }
         }
      }

      void fillHandles(edm::Run const& run, std::vector<edm::Handle<T> >& handles) const {
         handles.clear();
         handles.reserve(inputTags_->size());
         edm::Handle<T> handle;
         for (auto const& inputTag : *inputTags_) {
            run.getByLabel(inputTag, handle);
            if (handle.isValid()) {
               handles.push_back(handle);
            }
         }
      }

      std::vector<edm::InputTag> const& inputTags() const { return *inputTags_; }
      edm::BranchType branchType() const { return branchType_; }

   private:

      std::function<bool (edm::BranchDescription const& branchDescription)> matcher_;
      // A shared pointer is needed because objects of this type get assigned
      // to std::function's and we want the copies in those to share the same vector.
      std::shared_ptr<std::vector<edm::InputTag> > inputTags_;
      edm::BranchType branchType_;
   };
}
#endif
