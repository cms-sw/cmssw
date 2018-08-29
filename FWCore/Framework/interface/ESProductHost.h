#ifndef FWCore_Framework_ESProductHost_h
#define FWCore_Framework_ESProductHost_h
// -*- C++ -*-
//
// Package:     Framework
// Class:      ESProductHost
//
/**\class edm::ESProductHost

  Description: Helps an ESProducer produce an ESProduct that has
complex dependences on multiple record types. In particular,
it helps it manage what is executed in the produce function
based on which records changed and which did not.

This was designed as a replacement for the EventSetup "dependsOn"
functionality and is needed in some cases where the "dependsOn"
callbacks would not work well when processing multiple IOVs
concurrently.

  Usage:

An ESProducer class would use this class.

1. Add to the ESProducer header

    #include "FWCore/Framework/interface/ESProductHost.h"
    #include "FWCore/Utilities/interface/ReusableObjectHolder.h"

2. Assume the following only for purposes of this usage example
(the actual types in your case would be different). If the produced
function was defined as follows:

    std::shared_ptr<ESTestDataB> produce(ESTestRecordB const&);

and assume there is some special dependence on ESTestRecordC and
ESTestRecordD.

3. Then you would add a data member to the ESProducer similar
to the following:

    using HostType = edm::ESProductHost<ESTestDataB,
                                        ESTestRecordC,
                                        ESTestRecordD>;

    edm::ReusableObjectHolder<HostType> holder_;

4. Then the produce function would be defined something similar
to this:

    std::shared_ptr<ESTestDataB> ESTestProducerBUsingHost::produce(ESTestRecordB const& record) {

      auto host = holder_.makeOrGet([]() {
        return new HostType;
      });

      host->ifRecordChanges<ESTestRecordC>(record,
                                           [this, h=host.get()](auto const& rec) {
        // Do whatever special needs to be done only when record C changes
        // (If you are using this class to replace a dependsOn callback, you could
        // call the function that dependsOn used as a callback here)
      });

      host->ifRecordChanges<ESTestRecordD>(record,
                                           [this, h=host.get()](auto const& rec) {
        // Do whatever special needs to be done only when record D changes
      });

    ... repeat for as many record types as you need.
*/
//
// Author:      W. David Dagenhart
// Created:     28 August 2018
//

#include <cstddef>
#include <type_traits>
#include <vector>

namespace edm {

  // The parameter pack RecordTypes should contain all the
  // record types which you want to use when calling the
  // function "ifRecordChanges" as the first template parameter
  // to that function (the RecordType). The list of types in
  // RecordTypes is used only to size the vector of cacheIDs_ and
  // to establish an order of the types in RecordTypes. The
  // order is only used to establish a one to one correspondence
  // between cacheIDs_ and the types. The particular order
  // selected does not mattter, just that some order exists
  // so we know which cacheID corresponds to which type.

  template <typename Product, typename ... RecordTypes>
  class ESProductHost final : public Product {

  public:

    template <typename... Args>
    ESProductHost(Args&&... args) : Product(std::forward<Args>(args)...),
                                    cacheIDs_(numberOfRecordTypes(), 0) { }

    // Execute FUNC if the cacheIdentifier in the EventSetup RecordType
    // has changed since the last time we called FUNC for
    // this EventSetup product.

    template <typename RecordType, typename ContainingRecordType, typename FUNC>
    void ifRecordChanges(ContainingRecordType const& containingRecord,
                         FUNC func) {
      RecordType const& record = containingRecord. template getRecord<RecordType>();
      unsigned long long cacheIdentifier = record.cacheIdentifier();
      std::size_t iRecord = index<RecordType>();
      if (cacheIdentifier != cacheIDs_[iRecord]) {
        cacheIDs_[iRecord] = cacheIdentifier;
        func(record);
      }
    }

    // The rest of the functions are not intended for public use.
    // The only reason they are not private is that test code
    // uses them.

    // Return the number of record types for which we want to check
    // if the cache identifier changed.

    constexpr static std::size_t numberOfRecordTypes() { return sizeof...(RecordTypes); }

    // Return the index of a record type in the types in the parameter pack RecordTypes.
    // The first type in the template parameters after the Product type
    // will have an index of 0, the next 1, and so on.
    // (There must be at least one type after Product if you want to call the
    // "ifRecordChanges" function).

    template <typename U>
    constexpr static std::size_t index() {
      static_assert(numberOfRecordTypes() > 0, "no record types in ESProductHost");
      return indexLoop<0, U, RecordTypes...>();
    }

    template <std::size_t I, typename U, typename TST, typename... M>
    constexpr static std::size_t indexLoop() {
      if constexpr (std::is_same_v<U, TST>) {
        return I;
      } else {
        static_assert(I + 1 < numberOfRecordTypes(), "unknown record type passed to ESProductHost::index");
        return indexLoop<I+1, U, M...>();
      }
    }

  private:

    // Data member, this holds the cache identifiers from the record which
    // are used to determine whether the EventSetup cache for that record has
    // been updated since this Product was lasted updated.

    std::vector<unsigned long long> cacheIDs_;
  };
}
#endif
