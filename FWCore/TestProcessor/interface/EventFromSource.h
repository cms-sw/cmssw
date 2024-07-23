#ifndef FWCore_TestProcessor_EventFromSource_h
#define FWCore_TestProcessor_EventFromSource_h
// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     EventFromSource
//
/**\class EventFromSource EventFromSource.h "EventFromSource.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 30 Apr 2018 18:51:27 GMT
//

// system include files
#include <string>

// user include files
#include "FWCore/TestProcessor/interface/TestHandle.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

// forward declarations

namespace edm {
  class EventFromSourcePrincipal;

  namespace test {

    class EventFromSource {
    public:
      EventFromSource(EventPrincipal const& iPrincipal, edm::ServiceToken iToken)
          : principal_(&iPrincipal), token_(iToken) {}

      // ---------- const member functions ---------------------
      template <typename T>
      TestHandle<T> get(std::string const& iModule,
                        std::string const& iInstanceLabel,
                        std::string const& iProcess) const {
        ServiceRegistry::Operate operate(token_);

        auto h = principal_->getByLabel(
            edm::PRODUCT_TYPE, edm::TypeID(typeid(T)), iModule, iInstanceLabel, iProcess, nullptr, nullptr, nullptr);
        if (h.failedToGet()) {
          return TestHandle<T>(std::move(h.whyFailedFactory()));
        }
        void const* basicWrapper = h.wrapper();
        assert(basicWrapper);
        Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(basicWrapper);
        return TestHandle<T>(wrapper->product());
      }

      RunNumber_t run() const { return principal_->run(); }
      LuminosityBlockNumber_t luminosityBlock() const { return principal_->luminosityBlock(); }
      EventNumber_t event() const { return principal_->aux().event(); }
      EventAuxiliary const& aux() const { return principal_->aux(); }

    private:
      // ---------- member data --------------------------------
      EventPrincipal const* principal_;
      edm::ServiceToken token_;
    };
  }  // namespace test
}  // namespace edm

#endif
