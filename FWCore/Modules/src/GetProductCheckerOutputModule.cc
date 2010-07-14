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
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
   class ParameterSet;
   class GetProductCheckerOutputModule : public OutputModule {
   public:
      // We do not take ownership of passed stream.
      explicit GetProductCheckerOutputModule(ParameterSet const& pset);
      virtual ~GetProductCheckerOutputModule();
      static void fillDescriptions(ConfigurationDescriptions& descriptions);

   private:
      virtual void write(EventPrincipal const& e);
      virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&);
      virtual void writeRun(RunPrincipal const&);
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
   static void check(Principal const& p, std::string const& id) {
      for(Principal::const_iterator it = p.begin(), itEnd = p.end();
          it != itEnd;
          ++it) {
         if(*it) {
            BranchID branchID = (*it)->branchDescription().branchID();
            OutputHandle const oh = p.getForOutput(branchID, false);
            
            if(0 != oh.desc() && oh.desc()->branchID() != branchID) {
               throw cms::Exception("BranchIDMissMatch") << "While processing " << id << " request for BranchID " << branchID << " returned BranchID " << oh.desc()->branchID() << "\n";
            }
            
            TypeID tid((*it)->branchDescription().type().TypeInfo());
            size_t temp = 0;
            int tempCount = -1;
            BasicHandle bh = p.getByLabel(tid,
            (*it)->branchDescription().moduleLabel(),
            (*it)->branchDescription().productInstanceName(),
            (*it)->branchDescription().processName(),
            temp, tempCount);
            
            /*This doesn't appear to be an error, it just means the Product isn't available, which can be legitimate
            if(!bh.product()) {
               throw cms::Exception("GetByLabelFailure") << "While processing " << id << " getByLabel request for " << (*it)->productDescription().moduleLabel()
                  << " '" << (*it)->productDescription().productInstanceName() << "' " << (*it)->productDescription().processName() << " failed\n.";
            }*/
            if(0 != bh.provenance() && bh.provenance()->branchDescription().branchID() != branchID) {
               throw cms::Exception("BranchIDMissMatch") << "While processing " << id << " getByLabel request for " << (*it)->branchDescription().moduleLabel()
                  << " '" << (*it)->branchDescription().productInstanceName() << "' " << (*it)->branchDescription().processName()
                  << "\n should have returned BranchID " << branchID << " but returned BranchID " << bh.provenance()->branchDescription().branchID() << "\n";
            }
         }
      }
   }
   void GetProductCheckerOutputModule::write(EventPrincipal const& e) {
      std::ostringstream str;
      str << e.id();
      check(e, str.str());
   }
   void GetProductCheckerOutputModule::writeLuminosityBlock(LuminosityBlockPrincipal const& l) {
      std::ostringstream str;
      str << l.id();
      check(l, str.str());
   }
   void GetProductCheckerOutputModule::writeRun(RunPrincipal const& r) {
      std::ostringstream str;
      str << r.id();
      check(r, str.str());
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
