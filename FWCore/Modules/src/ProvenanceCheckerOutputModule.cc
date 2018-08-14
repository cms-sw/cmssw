// -*- C++ -*-
//
// Package:     Modules
// Class  :     ProvenanceCheckerOutputModule
//
// Implementation:
//     Checks the consistency of provenance stored in the framework
//
// Original Author:  Chris Jones
//         Created:  Thu Sep 11 19:24:13 EDT 2008
//

// system include files
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


// user include files

namespace edm {

   class ModuleCallingContext;
   class ParameterSet;

   class ProvenanceCheckerOutputModule : public OutputModule {
   public:
      // We do not take ownership of passed stream.
      explicit ProvenanceCheckerOutputModule(ParameterSet const& pset);
      ~ProvenanceCheckerOutputModule() override;
      static void fillDescriptions(ConfigurationDescriptions& descriptions);

   private:
      void write(EventForOutput const& e) override;
      void writeLuminosityBlock(LuminosityBlockForOutput const&) override {}
      void writeRun(RunForOutput const&) override {}
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
   ProvenanceCheckerOutputModule::ProvenanceCheckerOutputModule(ParameterSet const& pset) :
   OutputModule(pset)
   {
   }

// ProvenanceCheckerOutputModule::ProvenanceCheckerOutputModule(ProvenanceCheckerOutputModule const& rhs)
// {
//    // do actual copying here;
// }

   ProvenanceCheckerOutputModule::~ProvenanceCheckerOutputModule()
   {
   }

//
// assignment operators
//
// ProvenanceCheckerOutputModule const& ProvenanceCheckerOutputModule::operator=(ProvenanceCheckerOutputModule const& rhs)
// {
//   //An exception safe implementation is
//   ProvenanceCheckerOutputModule temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

   namespace {
     void markAncestors(EventForOutput const& e,
                             ProductProvenance const& iInfo,
                             std::map<BranchID, bool>& oMap,
                             std::set<BranchID>& oMapperMissing) {
       for(BranchID const id : iInfo.parentage().parents()) {
         //Don't look for parents if we've previously looked at the parents
         if(oMap.find(id) == oMap.end()) {
            //use side effect of calling operator[] which is if the item isn't there it will add it as 'false'
            oMap[id];
            ProductProvenance const* pInfo = e.getProvenance(id).productProvenance();
            if(pInfo) {
               markAncestors(e, *pInfo, oMap, oMapperMissing);
            } else {
               oMapperMissing.insert(id);
            }
         }
       }
     }
   }

   void
   ProvenanceCheckerOutputModule::write(EventForOutput const& e) {
      //check ProductProvenance's parents to see if they are in the ProductProvenance list

      std::map<BranchID, bool> seenParentInPrincipal;
      std::set<BranchID> missingFromMapper;
      std::set<BranchID> missingProductProvenance;

      std::map<BranchID, const BranchDescription*> idToBranchDescriptions;
      for(auto const product : keptProducts()[InEvent]) {
        BranchDescription const* branchDescription = product.first; 
        BranchID branchID = branchDescription->branchID();
        idToBranchDescriptions[branchID] = branchDescription;
        TypeID const& tid(branchDescription->unwrappedTypeID());
        EDGetToken const& token = product.second;
        BasicHandle bh;
        e.getByToken(token, tid, bh);
             bool cannotFindProductProvenance=false;
             if(!(bh.provenance() and bh.provenance()->productProvenance())) {
                missingProductProvenance.insert(branchID);
                cannotFindProductProvenance=true;
             }
             ProductProvenance const* pInfo = e.getProvenance(branchID).productProvenance();
             if(!pInfo) {
                missingFromMapper.insert(branchID);
                continue;
             }
             if(cannotFindProductProvenance) {
                continue;
             }
             markAncestors(e, *(bh.provenance()->productProvenance()), seenParentInPrincipal, missingFromMapper);
            seenParentInPrincipal[branchID] = true;
      }

      //Determine what BranchIDs are in the product registry
      Service<ConstProductRegistry> reg;
      ProductRegistry::ProductList const& prodList = reg->productList();
      std::set<BranchID> branchesInReg;
      for(auto const& product : prodList) {
         branchesInReg.insert(product.second.branchID());
         idToBranchDescriptions[product.second.branchID()] = &product.second;
      }

      std::set<BranchID> missingFromReg;
      for(auto const& item : seenParentInPrincipal) {
         if(branchesInReg.find(item.first) == branchesInReg.end()) {
            missingFromReg.insert(item.first);
         }
      }

      if(!missingFromMapper.empty()) {
         LogError("ProvenanceChecker") << "Missing the following BranchIDs from ProductProvenanceRetriever\n";
         for(std::set<BranchID>::iterator it = missingFromMapper.begin(), itEnd = missingFromMapper.end();
             it != itEnd;
             ++it) {
            LogProblem("ProvenanceChecker") << *it<<" "<<*(idToBranchDescriptions[*it]);
         }
      }

      if(!missingProductProvenance.empty()) {
         LogError("ProvenanceChecker") << "The ProductResolvers for the following BranchIDs have no ProductProvenance\n";
         for(std::set<BranchID>::iterator it = missingProductProvenance.begin(), itEnd = missingProductProvenance.end();
             it != itEnd;
             ++it) {
            LogProblem("ProvenanceChecker") << *it<<" "<<*(idToBranchDescriptions[*it]);
         }
      }

      if(!missingFromReg.empty()) {
         LogError("ProvenanceChecker") << "Missing the following BranchIDs from ProductRegistry\n";
         for(auto const& item : missingFromReg) {
            LogProblem("ProvenanceChecker") << item << " " << *(idToBranchDescriptions[item]);
         }
      }

      if(!missingFromMapper.empty() || !missingProductProvenance.empty() || !missingFromReg.empty()) {
         throw cms::Exception("ProvenanceError")
         << (!missingFromMapper.empty() ? "Having missing ancestors from ProductProvenanceRetriever.\n" : "")
         << (!missingProductProvenance.empty() ? " Have missing ProductProvenance's from ProductResolver in Event.\n" : "")
         << (!missingFromReg.empty() ? " Have missing info from ProductRegistry.\n" : "");
      }
   }

//
// const member functions
//

//
// static member functions
//
  void
  ProvenanceCheckerOutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    OutputModule::fillDescription(desc);
    descriptions.add("provenanceChecker", desc);
  }
}
using edm::ProvenanceCheckerOutputModule;
DEFINE_FWK_MODULE(ProvenanceCheckerOutputModule);
