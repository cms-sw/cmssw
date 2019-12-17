#ifndef DataFormats_FWLite_DataGetterHelper_h
#define DataFormats_FWLite_DataGetterHelper_h
// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     DataGetterHelper
//
/**\class DataGetterHelper DataGetterHelper.h src/DataFormats/FWLite/interface/DataGetterHelper.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author: Eric Vaandering
//         Created:  Fri Jan 29 12:45:17 CST 2010
//

// user include files
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/FWLite/interface/HistoryGetterBase.h"
#include "DataFormats/FWLite/interface/InternalDataKey.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "Rtypes.h"

// system include files
#include <cstring>
#include <map>
#include <memory>
#include <typeinfo>
#include <vector>
#include <functional>

// forward declarations
class TTreeCache;
class TTree;

namespace edm {
  class BranchDescription;
  class BranchID;
  class ObjectWithDict;
  class ProductID;
  class ThinnedAssociation;
  class WrapperBase;
}  // namespace edm

namespace fwlite {
  class DataGetterHelper {
  public:
    //            DataGetterHelper() {};
    DataGetterHelper(
        TTree* tree,
        std::shared_ptr<HistoryGetterBase> historyGetter,
        std::shared_ptr<BranchMapReader> branchMap = std::shared_ptr<BranchMapReader>(),
        std::shared_ptr<edm::EDProductGetter> getter = std::shared_ptr<edm::EDProductGetter>(),
        bool useCache = false,
        std::function<void(TBranch const&)> baFunc = [](TBranch const&) {});
    virtual ~DataGetterHelper();

    // ---------- const member functions ---------------------
    virtual std::string const getBranchNameFor(std::type_info const&, char const*, char const*, char const*) const;

    // This function should only be called by fwlite::Handle<>
    virtual bool getByLabel(std::type_info const&, char const*, char const*, char const*, void*, Long_t) const;

    edm::WrapperBase const* getByProductID(edm::ProductID const& pid, Long_t eventEntry) const;
    edm::WrapperBase const* getThinnedProduct(edm::ProductID const& pid, unsigned int& key, Long_t eventEntry) const;
    void getThinnedProducts(edm::ProductID const& pid,
                            std::vector<edm::WrapperBase const*>& foundContainers,
                            std::vector<unsigned int>& keys,
                            Long_t eventEntry) const;

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

    void setGetter(std::shared_ptr<edm::EDProductGetter const> getter) { getter_ = getter; }

    edm::EDProductGetter const* getter() const { return getter_.get(); }

  private:
    DataGetterHelper(const DataGetterHelper&) = delete;                   // stop default
    const DataGetterHelper& operator=(const DataGetterHelper&) = delete;  // stop default

    typedef std::map<internal::DataKey, std::shared_ptr<internal::Data>> KeyToDataMap;

    internal::Data& getBranchDataFor(std::type_info const&, char const*, char const*, char const*) const;
    void getBranchData(edm::EDProductGetter const*, Long64_t, internal::Data&) const;
    bool getByBranchDescription(edm::BranchDescription const&, Long_t eventEntry, KeyToDataMap::iterator&) const;
    edm::WrapperBase const* getByBranchID(edm::BranchID const& bid, Long_t eventEntry) const;
    edm::WrapperBase const* wrapperBasePtr(edm::ObjectWithDict const&) const;
    edm::ThinnedAssociation const* getThinnedAssociation(edm::BranchID const& branchID, Long_t eventEntry) const;

    // ---------- member data --------------------------------
    TTree* tree_;
    //This class is not inteded to be used across different threads
    CMS_SA_ALLOW mutable std::shared_ptr<BranchMapReader> branchMap_;
    CMS_SA_ALLOW mutable KeyToDataMap data_;
    CMS_SA_ALLOW mutable std::vector<char const*> labels_;
    const edm::ProcessHistory& history() const;

    CMS_SA_ALLOW mutable std::map<std::pair<edm::ProductID, edm::BranchListIndex>, std::shared_ptr<internal::Data>>
        idToData_;
    CMS_SA_ALLOW mutable std::map<edm::BranchID, std::shared_ptr<internal::Data>> bidToData_;
    edm::propagate_const<std::shared_ptr<fwlite::HistoryGetterBase>> historyGetter_;
    std::shared_ptr<edm::EDProductGetter const> getter_;
    CMS_SA_ALLOW mutable bool tcTrained_;
    /// Use internal TTreeCache.
    const bool tcUse_;
    /// Branch-access-function gets called whenever a branch data is accessed.
    /// This can be used for management of TTreeCache on the user side.
    std::function<void(TBranch const&)> branchAccessFunc_;
  };

}  // namespace fwlite

#endif
