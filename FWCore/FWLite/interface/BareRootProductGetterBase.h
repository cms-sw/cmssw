#ifndef FWCore_FWLite_BareRootProductGetterBase_h
#define FWCore_FWLite_BareRootProductGetterBase_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     BareRootProductGetterBase
//
/**\class BareRootProductGetterBase BareRootProductGetterBase.h FWCore/FWLite/interface/BareRootProductGetterBase.h

 Description: <one line class summary>

 This file was originally FWCore/FWLite/src/BareRootProductGetter.h,
 and was copied to the interface/BareRootProductGetterBase.h in order
 to refactor it a little bit to make it usable for FireworksWeb.

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
#include "FWCore/Utilities/interface/propagate_const.h"

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
}  // namespace edm

class BareRootProductGetterBase : public edm::EDProductGetter {
public:
  BareRootProductGetterBase();
  ~BareRootProductGetterBase() override;
  BareRootProductGetterBase(BareRootProductGetterBase const&) = delete;                   // stop default
  BareRootProductGetterBase const& operator=(BareRootProductGetterBase const&) = delete;  // stop default

  // ---------- const member functions ---------------------
  edm::WrapperBase const* getIt(edm::ProductID const&) const override;

private:
  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  unsigned int transitionIndex_() const override { return 0u; }

  edm::WrapperBase const* getIt(edm::BranchID const&, Long_t eventEntry) const;
  // This customization point was created for FireworksWeb
  virtual TFile* currentFile() const = 0;

  struct Buffer {
    Buffer(edm::WrapperBase const* iProd, TBranch* iBranch, void* iAddress, TClass* iClass)
        : product_(iProd), branch_(iBranch), address_(iAddress), eventEntry_(-1), class_(iClass) {}
    Buffer() : product_(), branch_(), address_(), eventEntry_(-1), class_(nullptr) {}

    std::shared_ptr<edm::WrapperBase const> product_;
    edm::propagate_const<TBranch*> branch_;
    void* address_;      //the address to pass to Root since as of 5.13 they cache that info
    Long_t eventEntry_;  //the event Entry used with the last GetEntry call
    edm::propagate_const<TClass*> class_;
  };

  Buffer* createNewBuffer(edm::BranchID const&) const;

  // ---------- member data --------------------------------

  typedef std::map<edm::BranchID, Buffer> IdToBuffers;
  mutable IdToBuffers idToBuffers_;
  mutable fwlite::BranchMapReader branchMap_;
};
#endif
