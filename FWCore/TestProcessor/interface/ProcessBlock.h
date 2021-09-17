#ifndef FWCore_TestProcessor_ProcessBlock_h
#define FWCore_TestProcessor_ProcessBlock_h
// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     ProcessBlock
//
/**\class edm::test::ProcessBlock

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  W. David Dagenhart
//         Created:  28 May 2020
//

#include <string>

#include "FWCore/TestProcessor/interface/TestHandle.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Utilities/interface/TypeID.h"

namespace edm {

  namespace test {

    class ProcessBlock {
    public:
      ProcessBlock(ProcessBlockPrincipal const* iPrincipal, std::string iModuleLabel, std::string iProcessName);

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

    private:
      ProcessBlockPrincipal const* principal_;
      std::string label_;
      std::string processName_;
    };
  }  // namespace test
}  // namespace edm

#endif
