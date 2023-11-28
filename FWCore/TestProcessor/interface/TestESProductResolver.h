#ifndef FWCore_TestProcessor_TestESProductResolver_h
#define FWCore_TestProcessor_TestESProductResolver_h
// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     TestESProductResolver
//
/**\class TestESProductResolver TestESProductResolver.h "TestESProductResolver.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Tue, 08 May 2018 18:32:38 GMT
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProductResolver.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

// forward declarations

namespace edm {
  namespace test {

    template <typename T>
    class TestESProductResolver : public eventsetup::ESProductResolver {
    public:
      TestESProductResolver() {}

      void setData(std::unique_ptr<T> iData) { data_ = std::move(iData); }

      void prefetchAsyncImpl(WaitingTaskHolder,
                             eventsetup::EventSetupRecordImpl const&,
                             eventsetup::DataKey const&,
                             EventSetupImpl const*,
                             ServiceToken const&,
                             ESParentContext const&) final {
        return;
      }

      void invalidateCache() final { data_.reset(); }

      void const* getAfterPrefetchImpl() const final { return data_.get(); }

    private:
      std::unique_ptr<T> data_;
    };
  }  // namespace test

}  // namespace edm

#endif
