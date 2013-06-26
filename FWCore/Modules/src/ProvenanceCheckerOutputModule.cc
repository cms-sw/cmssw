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
   class ParameterSet;
   class ProvenanceCheckerOutputModule : public OutputModule {
   public:
      // We do not take ownership of passed stream.
      explicit ProvenanceCheckerOutputModule(ParameterSet const& pset);
      virtual ~ProvenanceCheckerOutputModule();
      static void fillDescriptions(ConfigurationDescriptions& descriptions);

   private:
      virtual void write(EventPrincipal const& e);
      virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&){}
      virtual void writeRun(RunPrincipal const&){}
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
                             BranchMapper const& iMapper,
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
   ProvenanceCheckerOutputModule::write(EventPrincipal const& e) {
      //check ProductProvenance's parents to see if they are in the ProductProvenance list
      boost::shared_ptr<BranchMapper> mapperPtr = e.branchMapperPtr();

      std::map<BranchID, bool> seenParentInPrincipal;
      std::set<BranchID> missingFromMapper;
      std::set<BranchID> missingProductProvenance;

      std::map<BranchID, boost::shared_ptr<ProductHolderBase> > idToProductHolder;
      for(EventPrincipal::const_iterator it = e.begin(), itEnd = e.end();
          it != itEnd;
          ++it) {
        if(*it && (*it)->singleProduct()) {
            BranchID branchID = (*it)->branchDescription().branchID();
            idToProductHolder[branchID] = (*it);
            if((*it)->productUnavailable()) {
               //This call seems to have a side effect of filling the 'ProductProvenance' in the ProductHolder
               OutputHandle const oh = e.getForOutput(branchID, false);

               bool cannotFindProductProvenance=false;
               if(!(*it)->productProvenancePtr()) {
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
               markAncestors(*((*it)->productProvenancePtr()), *mapperPtr, seenParentInPrincipal, missingFromMapper);
            }
            seenParentInPrincipal[branchID] = true;
         }
      }

      //Determine what BranchIDs are in the product registry
      ProductRegistry const& reg = e.productRegistry();
      ProductRegistry::ProductList const prodList = reg.productList();
      std::set<BranchID> branchesInReg;
      for(ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
          it != itEnd;
          ++it) {
         branchesInReg.insert(it->second.branchID());
      }

      std::set<BranchID> missingFromPrincipal;
      std::set<BranchID> missingFromReg;
      for(std::map<BranchID, bool>::iterator it = seenParentInPrincipal.begin(), itEnd = seenParentInPrincipal.end();
          it != itEnd;
          ++it) {
         if(!it->second) {
            missingFromPrincipal.insert(it->first);
         }
         if(branchesInReg.find(it->first) == branchesInReg.end()) {
            missingFromReg.insert(it->first);
         }
      }

      if(missingFromMapper.size()) {
         LogError("ProvenanceChecker") << "Missing the following BranchIDs from BranchMapper\n";
         for(std::set<BranchID>::iterator it = missingFromMapper.begin(), itEnd = missingFromMapper.end();
             it != itEnd;
             ++it) {
            LogProblem("ProvenanceChecker") << *it<<" "<<idToProductHolder[*it]->branchDescription();
         }
      }
      if(missingFromPrincipal.size()) {
         LogError("ProvenanceChecker") << "Missing the following BranchIDs from EventPrincipal\n";
         for(std::set<BranchID>::iterator it = missingFromPrincipal.begin(), itEnd = missingFromPrincipal.end();
             it != itEnd;
             ++it) {
            LogProblem("ProvenanceChecker") << *it;
         }
      }

      if(missingProductProvenance.size()) {
         LogError("ProvenanceChecker") << "The ProductHolders for the following BranchIDs have no ProductProvenance\n";
         for(std::set<BranchID>::iterator it = missingProductProvenance.begin(), itEnd = missingProductProvenance.end();
             it != itEnd;
             ++it) {
            LogProblem("ProvenanceChecker") << *it<<" "<<idToProductHolder[*it]->branchDescription();
         }
      }

      if(missingFromReg.size()) {
         LogError("ProvenanceChecker") << "Missing the following BranchIDs from ProductRegistry\n";
         for(std::set<BranchID>::iterator it = missingFromReg.begin(), itEnd = missingFromReg.end();
             it != itEnd;
             ++it) {
            LogProblem("ProvenanceChecker") << *it;
         }
      }

      if(missingFromMapper.size() || missingFromPrincipal.size() || missingProductProvenance.size() || missingFromReg.size()) {
         throw cms::Exception("ProvenanceError")
         << (missingFromMapper.size() || missingFromPrincipal.size() ? "Having missing ancestors" : "")
         << (missingFromMapper.size() ? " from BranchMapper" : "")
         << (missingFromMapper.size() && missingFromPrincipal.size() ? " and" : "")
         << (missingFromPrincipal.size() ? " from EventPrincipal" : "")
         << (missingFromMapper.size() || missingFromPrincipal.size() ? ".\n" : "")
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
