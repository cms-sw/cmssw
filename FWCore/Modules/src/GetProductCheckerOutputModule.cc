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
#include "DataFormats/Common/interface/OutputHandle.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
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
      virtual ~GetProductCheckerOutputModule();
      static void fillDescriptions(ConfigurationDescriptions& descriptions);

   private:
      virtual void write(EventPrincipal const& e, edm::ModuleCallingContext const*) override;
      virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&, edm::ModuleCallingContext const*) override;
      virtual void writeRun(RunPrincipal const&, edm::ModuleCallingContext const*) override;
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
   GetProductCheckerOutputModule::GetProductCheckerOutputModule(ParameterSet const& iPSet):
   OutputModule(iPSet) { 
   } 

// GetProductCheckerOutputModule::GetProductCheckerOutputModule(GetProductCheckerOutputModule const& rhs) {
//    // do actual copying here;
// }

   GetProductCheckerOutputModule::~GetProductCheckerOutputModule() {
   }

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
   static void check(Principal const& p, std::string const& id, SelectedProducts const& iProducts, edm::ModuleCallingContext const* mcc) {
     for(auto const& branchDescription : iProducts) {
            TypeID const& tid(branchDescription->unwrappedTypeID());
            BasicHandle bh = p.getByLabel(PRODUCT_TYPE, tid,
                                          branchDescription->moduleLabel(),
                                          branchDescription->productInstanceName(),
                                          branchDescription->processName(),
                                          nullptr, nullptr, mcc);
            
        if(0 != bh.provenance() && bh.provenance()->branchDescription().branchID() != branchDescription->branchID()) {
           throw cms::Exception("BranchIDMissMatch") << "While processing " << id << " getByLabel request for " << branchDescription->moduleLabel()
              << " '" << branchDescription->productInstanceName() << "' " << branchDescription->processName()
              << "\n should have returned BranchID " << branchDescription->branchID() << " but returned BranchID " << bh.provenance()->branchDescription().branchID() << "\n";
        }
       
      }
   }
   void GetProductCheckerOutputModule::write(EventPrincipal const& e, edm::ModuleCallingContext const* mcc) {
      std::ostringstream str;
      str << e.id();
      check(e, str.str(), keptProducts()[InEvent], mcc);
   }
   void GetProductCheckerOutputModule::writeLuminosityBlock(LuminosityBlockPrincipal const& l, edm::ModuleCallingContext const* mcc) {
      std::ostringstream str;
      str << l.id();
      check(l, str.str(), keptProducts()[InLumi], mcc);
   }
   void GetProductCheckerOutputModule::writeRun(RunPrincipal const& r, edm::ModuleCallingContext const* mcc) {
      std::ostringstream str;
      str << r.id();
      check(r, str.str(), keptProducts()[InRun], mcc);
   }

//
// const member functions
//

//
// static member functions
//

  void
  GetProductCheckerOutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    OutputModule::fillDescription(desc);
    descriptions.add("productChecker", desc);
  }
}

using edm::GetProductCheckerOutputModule;
DEFINE_FWK_MODULE(GetProductCheckerOutputModule);
