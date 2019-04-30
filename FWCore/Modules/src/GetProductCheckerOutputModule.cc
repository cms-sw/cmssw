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
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
  class ModuleCallingContext;
  class ParameterSet;

  class GetProductCheckerOutputModule : public OutputModule {
  public:
    // We do not take ownership of passed stream.
    explicit GetProductCheckerOutputModule(ParameterSet const& pset);
    ~GetProductCheckerOutputModule() override;
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void write(EventForOutput const& e) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const&) override;
    void writeRun(RunForOutput const&) override;
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
  GetProductCheckerOutputModule::GetProductCheckerOutputModule(ParameterSet const& iPSet) : OutputModule(iPSet) {}

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
  static void check(T const& p, std::string const& id, SelectedProducts const& iProducts) {
    for (auto const& product : iProducts) {
      BranchDescription const* branchDescription = product.first;
      TypeID const& tid = branchDescription->unwrappedTypeID();
      EDGetToken const& token = product.second;
      BasicHandle bh = p.getByToken(token, tid);
      if (nullptr != bh.provenance() &&
          bh.provenance()->branchDescription().branchID() != branchDescription->branchID()) {
        throw cms::Exception("BranchIDMissMatch")
            << "While processing " << id << " getByToken request for " << branchDescription->moduleLabel() << " '"
            << branchDescription->productInstanceName() << "' " << branchDescription->processName()
            << "\n should have returned BranchID " << branchDescription->branchID() << " but returned BranchID "
            << bh.provenance()->branchDescription().branchID() << "\n";
      }
    }
  }
  void GetProductCheckerOutputModule::write(EventForOutput const& e) {
    std::ostringstream str;
    str << e.id();
    check(e, str.str(), keptProducts()[InEvent]);
  }
  void GetProductCheckerOutputModule::writeLuminosityBlock(LuminosityBlockForOutput const& l) {
    std::ostringstream str;
    str << l.id();
    check(l, str.str(), keptProducts()[InLumi]);
  }
  void GetProductCheckerOutputModule::writeRun(RunForOutput const& r) {
    std::ostringstream str;
    str << r.id();
    check(r, str.str(), keptProducts()[InRun]);
  }

  //
  // const member functions
  //

  //
  // static member functions
  //

  void GetProductCheckerOutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    OutputModule::fillDescription(desc);
    descriptions.add("productChecker", desc);
  }
}  // namespace edm

using edm::GetProductCheckerOutputModule;
DEFINE_FWK_MODULE(GetProductCheckerOutputModule);
