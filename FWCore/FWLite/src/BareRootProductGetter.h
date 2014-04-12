#ifndef FWLite_BareRootProductGetter_h
#define FWLite_BareRootProductGetter_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     BareRootProductGetter
//
/**\class BareRootProductGetter BareRootProductGetter.h FWCore/FWLite/interface/BareRootProductGetter.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue May 23 11:03:27 EDT 2006
//

// user include files
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Common/interface/WrapperOwningHolder.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"

// system include files
#include "Rtypes.h"
#include <map>

// forward declarations
class TBranch;
class TFile;
class TTree;

class BareRootProductGetter : public edm::EDProductGetter {

   public:
      BareRootProductGetter();
      virtual ~BareRootProductGetter();

      // ---------- const member functions ---------------------
      virtual edm::WrapperHolder getIt(edm::ProductID const&) const override;

private:

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual unsigned int transitionIndex_() const override {
        return 0u;
      }

      struct Buffer {
        Buffer(edm::WrapperOwningHolder const& iProd, TBranch* iBranch, void* iAddress,
               TClass* iClass) :
        product_(iProd), branch_(iBranch), address_(iAddress), eventEntry_(-1),
        class_(iClass) {}
        Buffer() : product_(), branch_(), address_(), eventEntry_(-1),class_(0) {}

        edm::WrapperOwningHolder product_;
        TBranch* branch_;
        void* address_; //the address to pass to Root since as of 5.13 they cache that info
        Long_t eventEntry_; //the event Entry used with the last GetEntry call
        TClass* class_;
      };
   private:
      BareRootProductGetter(BareRootProductGetter const&); // stop default

      BareRootProductGetter const& operator=(BareRootProductGetter const&); // stop default

      // ---------- member data --------------------------------
      void setupNewFile(TFile*) const;
      TBranch* findBranch(edm::ProductID const&) const;
      Buffer* createNewBuffer(edm::ProductID const&) const;

//      mutable TFile* presentFile_;
//      mutable TTree* eventTree_;
//      mutable Long_t eventEntry_;
//      typedef std::map<edm::ProductID,edm::BranchDescription> IdToBranchDesc;
//      mutable IdToBranchDesc idToBranchDesc_;
      typedef std::map<edm::ProductID, Buffer> IdToBuffers;
      mutable IdToBuffers idToBuffers_;
      mutable fwlite::BranchMapReader branchMap_;
};

#endif
