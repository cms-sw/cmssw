#ifndef FWCore_TestProcessor_LuminosityBlock_h
#define FWCore_TestProcessor_LuminosityBlock_h
// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     LuminosityBlock
//
/**\class LuminosityBlock LuminosityBlock.h "LuminosityBlock.h"

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
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Utilities/interface/TypeID.h"

// forward declarations

namespace edm {

  namespace test {

    class LuminosityBlock {
    public:
      LuminosityBlock(std::shared_ptr<LuminosityBlockPrincipal const> iPrincipal,
                      std::string iModuleLabel,
                      std::string iProcessName);

      // ---------- const member functions ---------------------
      template <typename T>
      TestHandle<T> get() const {
        static const std::string s_null;
        return get<T>(s_null);
      }

      template <typename T>
      TestHandle<T> get(std::string const& iInstanceLabel) const {
        auto h = principal_->getByLabel(
            edm::PRODUCT_TYPE, edm::TypeID(typeid(T)), label_, iInstanceLabel, processName_, nullptr, nullptr, nullptr);
        if (h.failedToGet()) {
          return TestHandle<T>(std::move(h.whyFailedFactory()));
        }
        void const* basicWrapper = h.wrapper();
        assert(basicWrapper);
        Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(basicWrapper);
        return TestHandle<T>(wrapper->product());
      }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

    private:
      // ---------- member data --------------------------------
      std::shared_ptr<LuminosityBlockPrincipal const> principal_;
      std::string label_;
      std::string processName_;
    };
  }  // namespace test
}  // namespace edm

#endif
