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
// $Id: BareRootProductGetter.h,v 1.8 2008/06/03 17:35:03 dsr Exp $
//

// system include files
#include <map>
#include "Rtypes.h"

// user include files
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"

// forward declarations
class TBranch;
class TFile;
class TTree;

class BareRootProductGetter : public edm::EDProductGetter
{

   public:
      BareRootProductGetter();
      virtual ~BareRootProductGetter();

      // ---------- const member functions ---------------------
      virtual edm::EDProduct const* getIt(edm::ProductID const&) const;
      
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      struct Buffer {
        Buffer(edm::EDProduct* iProd, TBranch* iBranch, void* iAddress,
               TClass* iClass) :
        product_(iProd), branch_(iBranch), address_(iAddress), eventEntry_(-1),
        class_(iClass) {}
        Buffer() : product_(), branch_(), address_(), eventEntry_(-1),class_(0) {}
        
        boost::shared_ptr<edm::EDProduct const> product_;
        TBranch* branch_;
        void* address_; //the address to pass to Root since as of 5.13 they cache that info
        Long_t eventEntry_; //the event Entry used with the last GetEntry call
        TClass* class_;
      };
   private:
      BareRootProductGetter(const BareRootProductGetter&); // stop default

      const BareRootProductGetter& operator=(const BareRootProductGetter&); // stop default

      // ---------- member data --------------------------------
      void setupNewFile(TFile*) const;
      TBranch* findBranch(const edm::ProductID& ) const;
      Buffer* createNewBuffer(const edm::ProductID& ) const;
      
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
