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
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/OutputHandle.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// user include files

namespace edm {

   class ModuleCallingContext;
   class ParameterSet;

   class ProvenanceCheckerOutputModule : public OutputModule {
   public:
      // We do not take ownership of passed stream.
      explicit ProvenanceCheckerOutputModule(ParameterSet const& pset);
      virtual ~ProvenanceCheckerOutputModule();
      static void fillDescriptions(ConfigurationDescriptions& descriptions);

   private:
      virtual void write(EventPrincipal const& e, ModuleCallingContext const*) override;
      virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&, ModuleCallingContext const*) override {}
      virtual void writeRun(RunPrincipal const&, ModuleCallingContext const*) override {}
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
     void markAncestors(ProductProvenance const& iInfo,
                             ProductProvenanceRetriever const& iMapper,
                             std::map<BranchID, bool>& oMap,
                             std::set<BranchID>& oMapperMissing) {
       for(std::vector<BranchID>::const_iterator it = iInfo.parentage().parents().begin(),
          itEnd = iInfo.parentage().parents().end();
          it != itEnd;
          ++it) {
         //Don't look for parents if we've previously looked at the parents
         if(oMap.find(*it) == oMap.end()) {
            //use side effect of calling operator[] which is if the item isn't there it will add it as 'false'
            oMap[*it];
            ProductProvenance const* pInfo = iMapper.branchIDToProvenance(*it);
            if(pInfo) {
               markAncestors(*pInfo, iMapper, oMap, oMapperMissing);
            } else {
               oMapperMissing.insert(*it);
            }
         }
       }
     }
   }

   void
   ProvenanceCheckerOutputModule::write(EventPrincipal const& e, ModuleCallingContext const* mcc) {
      //check ProductProvenance's parents to see if they are in the ProductProvenance list
      auto mapperPtr = e.productProvenanceRetrieverPtr();

      std::map<BranchID, bool> seenParentInPrincipal;
      std::set<BranchID> missingFromMapper;
      std::set<BranchID> missingProductProvenance;

      std::map<BranchID, const BranchDescription*> idToBranchDescriptions;
      for(auto const branchDescription : keptProducts()[InEvent]) {
        BranchID branchID = branchDescription->branchID();
        idToBranchDescriptions[branchID] = branchDescription;

        TypeID const& tid(branchDescription->unwrappedTypeID());
        BasicHandle bh = e.getByLabel(PRODUCT_TYPE, tid,
                                      branchDescription->moduleLabel(),
                                      branchDescription->productInstanceName(),
                                      branchDescription->processName(),
                                      nullptr, nullptr, mcc);

             bool cannotFindProductProvenance=false;
             if(!(bh.provenance() and bh.provenance()->productProvenance())) {
                missingProductProvenance.insert(branchID);
                cannotFindProductProvenance=true;
             }
             ProductProvenance const* pInfo = mapperPtr->branchIDToProvenance(branchID);
             if(!pInfo) {
                missingFromMapper.insert(branchID);
                continue;
             }
             if(cannotFindProductProvenance) {
                continue;
             }
             markAncestors(*(bh.provenance()->productProvenance()), *mapperPtr, seenParentInPrincipal, missingFromMapper);
            seenParentInPrincipal[branchID] = true;
      }

      //Determine what BranchIDs are in the product registry
      ProductRegistry const& reg = e.productRegistry();
      ProductRegistry::ProductList const prodList = reg.productList();
      std::set<BranchID> branchesInReg;
      for(ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
          it != itEnd;
          ++it) {
         branchesInReg.insert(it->second.branchID());
         idToBranchDescriptions[it->second.branchID()] = &(it->second);
      }

      std::set<BranchID> missingFromReg;
      for(std::map<BranchID, bool>::iterator it = seenParentInPrincipal.begin(), itEnd = seenParentInPrincipal.end();
          it != itEnd;
          ++it) {
         if(branchesInReg.find(it->first) == branchesInReg.end()) {
            missingFromReg.insert(it->first);
         }
      }

      if(missingFromMapper.size()) {
         LogError("ProvenanceChecker") << "Missing the following BranchIDs from ProductProvenanceRetriever\n";
         for(std::set<BranchID>::iterator it = missingFromMapper.begin(), itEnd = missingFromMapper.end();
             it != itEnd;
             ++it) {
            LogProblem("ProvenanceChecker") << *it<<" "<<*(idToBranchDescriptions[*it]);
         }
      }

      if(missingProductProvenance.size()) {
         LogError("ProvenanceChecker") << "The ProductHolders for the following BranchIDs have no ProductProvenance\n";
         for(std::set<BranchID>::iterator it = missingProductProvenance.begin(), itEnd = missingProductProvenance.end();
             it != itEnd;
             ++it) {
            LogProblem("ProvenanceChecker") << *it<<" "<<*(idToBranchDescriptions[*it]);
         }
      }

      if(missingFromReg.size()) {
         LogError("ProvenanceChecker") << "Missing the following BranchIDs from ProductRegistry\n";
         for(std::set<BranchID>::iterator it = missingFromReg.begin(), itEnd = missingFromReg.end();
             it != itEnd;
             ++it) {
            LogProblem("ProvenanceChecker") << *it<<" "<<*(idToBranchDescriptions[*it]);
         }
      }

      if(missingFromMapper.size() || missingProductProvenance.size() || missingFromReg.size()) {
         throw cms::Exception("ProvenanceError")
         << (missingFromMapper.size() ? "Having missing ancestors from ProductProvenanceRetriever.\n" : "")
         << (missingProductProvenance.size() ? " Have missing ProductProvenance's from ProductHolder in EventPrincipal.\n" : "")
         << (missingFromReg.size() ? " Have missing info from ProductRegistry.\n" : "");
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
