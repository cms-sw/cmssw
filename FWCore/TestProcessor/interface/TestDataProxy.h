#ifndef FWCore_TestProcessor_TestDataProxy_h
#define FWCore_TestProcessor_TestDataProxy_h
// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     TestDataProxy
//
/**\class TestDataProxy TestDataProxy.h "TestDataProxy.h"

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
#include "FWCore/Framework/interface/DataProxy.h"

// forward declarations

namespace edm {
  namespace test {

    template <typename T>
    class TestDataProxy : public eventsetup::DataProxy {
    public:
      TestDataProxy() {}

      void setData(std::unique_ptr<T> iData) { data_ = std::move(iData); }

      void const* getImpl(eventsetup::EventSetupRecordImpl const&,
                          eventsetup::DataKey const&,
                          EventSetupImpl const*) final {
        return data_.get();
      }

      void invalidateCache() final { data_.reset(); }

    private:
      std::unique_ptr<T> data_;
    };
  }  // namespace test

}  // namespace edm

#endif
