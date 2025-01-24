// -*- C++ -*-
//
// Package:     Modules
// Class  :     GetProductCheckerOutputModule
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Wed Oct  7 14:41:26 CDT 2009
//

// system include files
#include <string>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {
  class ModuleCallingContext;
  class ParameterSet;

  class GetProductCheckerOutputModule : public one::OutputModule<> {
  public:
    // We do not take ownership of passed stream.
    explicit GetProductCheckerOutputModule(ParameterSet const& pset);
    ~GetProductCheckerOutputModule() override;
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void write(EventForOutput const& e) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const&) override;
    void writeRun(RunForOutput const&) override;
    const std::vector<std::string> crosscheck_;
    const bool verbose_;
  };

  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  GetProductCheckerOutputModule::GetProductCheckerOutputModule(ParameterSet const& iPSet)
      : one::OutputModuleBase(iPSet),
        one::OutputModule<>(iPSet),
        crosscheck_(iPSet.getUntrackedParameter<std::vector<std::string>>("crosscheck")),
        verbose_(iPSet.getUntrackedParameter<bool>("verbose")) {}

  // GetProductCheckerOutputModule::GetProductCheckerOutputModule(GetProductCheckerOutputModule const& rhs) {
  //    // do actual copying here;
  // }

  GetProductCheckerOutputModule::~GetProductCheckerOutputModule() {}

  //
  // assignment operators
  //
  // GetProductCheckerOutputModule const& GetProductCheckerOutputModule::operator=(GetProductCheckerOutputModule const& rhs) {
  //   //An exception safe implementation is
  //   GetProductCheckerOutputModule temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //
  template <typename T>
  static void check(T const& p, std::string const& id, SelectedProducts const& iProducts, bool iVerbose) {
    for (auto const& product : iProducts) {
      ProductDescription const* productDescription = product.first;
      TypeID const& tid = productDescription->unwrappedTypeID();
      EDGetToken const& token = product.second;
      BasicHandle bh = p.getByToken(token, tid);
      if (iVerbose) {
        if (bh.isValid()) {
          edm::LogInfo("FoundProduct") << "found " << productDescription->moduleLabel() << " '"
                                       << productDescription->productInstanceName() << "' "
                                       << productDescription->processName();
        } else {
          edm::LogInfo("DidNotFindProduct")
              << "did not find " << productDescription->moduleLabel() << " '" << productDescription->productInstanceName()
              << "' " << productDescription->processName();
        }
      }
      if (nullptr != bh.provenance() &&
          bh.provenance()->productDescription().branchID() != productDescription->branchID()) {
        throw cms::Exception("BranchIDMissMatch")
            << "While processing " << id << " getByToken request for " << productDescription->moduleLabel() << " '"
            << productDescription->productInstanceName() << "' " << productDescription->processName()
            << "\n should have returned BranchID " << productDescription->branchID() << " but returned BranchID "
            << bh.provenance()->productDescription().branchID() << "\n";
      }
    }
  }
  namespace {
    std::string canonicalName(std::string const& iOriginal) {
      if (iOriginal.empty()) {
        return iOriginal;
      }
      if (iOriginal.back() == '.') {
        return iOriginal.substr(0, iOriginal.size() - 1);
      }
      return iOriginal;
    }
  }  // namespace
  void GetProductCheckerOutputModule::write(EventForOutput const& e) {
    std::ostringstream str;
    str << e.id();
    check(e, str.str(), keptProducts()[InEvent], verbose_);
    if (not crosscheck_.empty()) {
      std::set<std::string> expectedProducts(crosscheck_.begin(), crosscheck_.end());
      for (auto const& kp : keptProducts()[InEvent]) {
        auto bn = canonicalName(kp.first->branchName());
        auto found = expectedProducts.find(bn);
        if (found == expectedProducts.end()) {
          throw cms::Exception("CrosscheckFailed") << "unexpected kept product " << bn;
        }
        expectedProducts.erase(bn);
      }
      if (not expectedProducts.empty()) {
        cms::Exception e("CrosscheckFailed");
        e << "Did not find the expected products:\n";
        for (auto const& p : expectedProducts) {
          e << p << "\n";
        }
        throw e;
      }
    }
  }
  void GetProductCheckerOutputModule::writeLuminosityBlock(LuminosityBlockForOutput const& l) {
    std::ostringstream str;
    str << l.id();
    check(l, str.str(), keptProducts()[InLumi], verbose_);
  }
  void GetProductCheckerOutputModule::writeRun(RunForOutput const& r) {
    std::ostringstream str;
    str << r.id();
    check(r, str.str(), keptProducts()[InRun], verbose_);
  }

  //
  // const member functions
  //

  //
  // static member functions
  //

  void GetProductCheckerOutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    one::OutputModule<>::fillDescription(desc);
    desc.addUntracked<std::vector<std::string>>("crosscheck", {})
        ->setComment("Branch names that should be in the event. If empty no check done.");
    desc.addUntracked<bool>("verbose", false);
    descriptions.add("productChecker", desc);
  }
}  // namespace edm

using edm::GetProductCheckerOutputModule;
DEFINE_FWK_MODULE(GetProductCheckerOutputModule);
