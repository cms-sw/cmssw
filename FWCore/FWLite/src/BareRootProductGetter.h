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
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"

// system include files
#include "Rtypes.h"
#include <map>
#include <memory>
#include <vector>

// forward declarations
class TBranch;
class TClass;

namespace edm {
   class BranchID;
   class ProductID;
   class ThinnedAssociation;
}

class BareRootProductGetter : public edm::EDProductGetter {

   public:
      BareRootProductGetter();
      virtual ~BareRootProductGetter();

      // ---------- const member functions ---------------------
      virtual edm::WrapperBase const* getIt(edm::ProductID const&) const override;

     // getThinnedProduct assumes getIt was already called and failed to find
     // the product. The input key is the index of the desired element in the
     // container identified by ProductID (which cannot be found).
     // If the return value is not null, then the desired element was found
     // in a thinned container and key is modified to be the index into
     // that thinned container. If the desired element is not found, then
     // nullptr is returned.
      virtual edm::WrapperBase const* getThinnedProduct(edm::ProductID const&,
                                                        unsigned int& key) const override;

     // getThinnedProducts assumes getIt was already called and failed to find
     // the product. The input keys are the indexes into the container identified
     // by ProductID (which cannot be found). On input the WrapperBase pointers
     // must all be set to nullptr (except when the function calls itself
     // recursively where non-null pointers mark already found elements).
     // Thinned containers derived from the product are searched to see
     // if they contain the desired elements. For each that is
     // found, the corresponding WrapperBase pointer is set and the key
     // is modified to be the key into the container where the element
     // was found. The WrapperBase pointers might or might not all point
     // to the same thinned container.
      virtual void getThinnedProducts(edm::ProductID const&,
                                      std::vector<edm::WrapperBase const*>& foundContainers,
                                      std::vector<unsigned int>& keys) const override;

   private:

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual unsigned int transitionIndex_() const override {
        return 0u;
      }

      edm::WrapperBase const* getIt(edm::BranchID const&, Long_t eventEntry) const;

      struct Buffer {
        Buffer(edm::WrapperBase const* iProd, TBranch* iBranch, void* iAddress,
               TClass* iClass) :
        product_(iProd), branch_(iBranch), address_(iAddress), eventEntry_(-1),
        class_(iClass) {}
        Buffer() : product_(), branch_(), address_(), eventEntry_(-1), class_(nullptr) {}

        std::shared_ptr<edm::WrapperBase const> product_;
        TBranch* branch_;
        void* address_; //the address to pass to Root since as of 5.13 they cache that info
        Long_t eventEntry_; //the event Entry used with the last GetEntry call
        TClass* class_;
      };

      BareRootProductGetter(BareRootProductGetter const&); // stop default

      BareRootProductGetter const& operator=(BareRootProductGetter const&); // stop default

      Buffer* createNewBuffer(edm::BranchID const&) const;
      edm::ThinnedAssociation const* getThinnedAssociation(edm::BranchID const& branchID, Long_t eventEntry) const;

      // ---------- member data --------------------------------

      typedef std::map<edm::BranchID, Buffer> IdToBuffers;
      mutable IdToBuffers idToBuffers_;
      mutable fwlite::BranchMapReader branchMap_;
};
#endif
