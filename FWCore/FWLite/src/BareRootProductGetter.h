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
// $Id: BareRootProductGetter.h,v 1.2 2006/08/21 20:56:20 wmtan Exp $
//

// system include files
#include <map>
#include "boost/shared_ptr.hpp"
#include "Rtypes.h"
#include "TUUID.h"

// user include files
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/EDProduct.h"

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
        Buffer(edm::EDProduct* iProd, TBranch* iBranch) :
        product_(iProd), branch_(iBranch) {}
        Buffer() : product_(), branch_() {}
        
        boost::shared_ptr<edm::EDProduct const> product_;
        TBranch* branch_;
      };
   private:
      BareRootProductGetter(const BareRootProductGetter&); // stop default

      const BareRootProductGetter& operator=(const BareRootProductGetter&); // stop default

      // ---------- member data --------------------------------
      void setupNewFile(TFile*) const;
      TBranch* findBranch(const edm::ProductID& ) const;
      Buffer* createNewBuffer(const edm::ProductID& ) const;
      
      mutable TFile* presentFile_;
      mutable TTree* eventTree_;
      mutable Long_t eventEntry_;
      typedef std::map<edm::ProductID,edm::BranchDescription> IdToBranchDesc;
      mutable IdToBranchDesc idToBranchDesc_;
      typedef std::map<edm::ProductID, Buffer> IdToBuffers;
      mutable IdToBuffers idToBuffers_;
      mutable TUUID fileUUID_;
      
};


#endif
