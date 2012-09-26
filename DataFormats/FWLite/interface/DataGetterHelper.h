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

#if !defined(__CINT__) && !defined(__MAKECINT__)

// user include files
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/FWLite/interface/HistoryGetterBase.h"
#include "DataFormats/FWLite/interface/InternalDataKey.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"

// system include files
#include "boost/shared_ptr.hpp"

#include <cstring>
#include <map>
#include <memory>
#include <typeinfo>
#include <vector>

// forward declarations
class TTreeCache;
class TTree;

namespace fwlite {
    class DataGetterHelper {

        public:
//            DataGetterHelper() {};
            DataGetterHelper(TTree* tree,
                             boost::shared_ptr<HistoryGetterBase> historyGetter,
                             boost::shared_ptr<BranchMapReader> branchMap = boost::shared_ptr<BranchMapReader>(),
                             boost::shared_ptr<edm::EDProductGetter> getter = boost::shared_ptr<edm::EDProductGetter>(),
                             bool useCache = false);
            virtual ~DataGetterHelper();

            // ---------- const member functions ---------------------
            virtual std::string const getBranchNameFor(std::type_info const&,
                                                        char const*,
                                                        char const*,
                                                        char const*) const;

            // This function should only be called by fwlite::Handle<>
            virtual bool getByLabel(std::type_info const&, char const*, char const*, char const*, void*, Long_t) const;
            virtual bool getByLabel(std::type_info const&, char const*, char const*, char const*, edm::WrapperHolder&, Long_t) const;
            edm::WrapperHolder getByProductID(edm::ProductID const&, Long_t) const;

            // ---------- static member functions --------------------
            static void throwProductNotFoundException(std::type_info const&, char const*, char const*, char const*);

            // ---------- member functions ---------------------------

            void setGetter(boost::shared_ptr<edm::EDProductGetter> getter) {
                getter_ = getter;
            }

            edm::EDProductGetter* getter() {
               return getter_.get();
            }

        private:
            DataGetterHelper(const DataGetterHelper&); // stop default

            const DataGetterHelper& operator=(const DataGetterHelper&); // stop default
            internal::Data& getBranchDataFor(std::type_info const&, char const*, char const*, char const*) const;
            void getBranchData(edm::EDProductGetter*, Long64_t, internal::Data&) const;

            // ---------- member data --------------------------------
            TTree* tree_;
            mutable boost::shared_ptr<BranchMapReader> branchMap_;
            typedef std::map<internal::DataKey, boost::shared_ptr<internal::Data> > KeyToDataMap;
            mutable KeyToDataMap data_;
            mutable std::vector<char const*> labels_;
            const edm::ProcessHistory& history() const;

            mutable std::map<std::pair<edm::ProductID, edm::BranchListIndexes>,boost::shared_ptr<internal::Data> > idToData_;
            boost::shared_ptr<fwlite::HistoryGetterBase> historyGetter_;
            boost::shared_ptr<edm::EDProductGetter> getter_;
            mutable std::auto_ptr<TTreeCache> tcache_;
            mutable bool tcTrained_;
    };

}

#endif /*__CINT__ */

#endif
