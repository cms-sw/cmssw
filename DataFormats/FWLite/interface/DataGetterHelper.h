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
// $Id: DataGetterHelper.h,v 1.5 2010/03/23 16:19:21 chrjones Exp $
//

#if !defined(__CINT__) && !defined(__MAKECINT__)

// system include files
#include <typeinfo>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <memory>
#include <cstring>

// user include files
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/FWLite/interface/InternalDataKey.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"
#include "DataFormats/FWLite/interface/HistoryGetterBase.h"

// forward declarations
class TTreeCache;
class TTree;

namespace fwlite {
    class DataGetterHelper
    {

        public:
//            DataGetterHelper() {};
            DataGetterHelper(TTree* tree,
                             boost::shared_ptr<HistoryGetterBase> historyGetter,
                             boost::shared_ptr<BranchMapReader> branchMap = boost::shared_ptr<BranchMapReader>(),
                             boost::shared_ptr<edm::EDProductGetter> getter = boost::shared_ptr<edm::EDProductGetter>(),
                             bool useCache = false);
            virtual ~DataGetterHelper();

            // ---------- const member functions ---------------------
            virtual const std::string getBranchNameFor(const std::type_info&,
                                                        const char*,
                                                        const char*,
                                                        const char*) const;

            // This function should only be called by fwlite::Handle<>
            virtual bool getByLabel(const std::type_info&, const char*, const char*, const char*, void*, Long_t) const;
            edm::EDProduct const* getByProductID(edm::ProductID const&, Long_t) const;

            // ---------- static member functions --------------------
            static void throwProductNotFoundException(const std::type_info&, const char*, const char*, const char*);

            // ---------- member functions ---------------------------

            void setGetter( boost::shared_ptr<edm::EDProductGetter> getter ) {
                getter_ = getter;
            }
       
            edm::EDProductGetter* getter() {
               return getter_.get();
            }

        private:
            DataGetterHelper(const DataGetterHelper&); // stop default

            const DataGetterHelper& operator=(const DataGetterHelper&); // stop default
            TTree* tree_;
            Long64_t index_;
            internal::Data& getBranchDataFor(const std::type_info&, const char*, const char*, const char*) const;
            void getBranchData(edm::EDProductGetter*, Long64_t, internal::Data&) const;

            // ---------- member data --------------------------------
            mutable boost::shared_ptr<BranchMapReader> branchMap_;
            // ---------- member data --------------------------------
            typedef std::map<internal::DataKey, boost::shared_ptr<internal::Data> > KeyToDataMap;
            mutable KeyToDataMap data_;
            mutable std::vector<const char*> labels_;
            const edm::ProcessHistory& history() const;

            mutable std::map<edm::ProductID,boost::shared_ptr<internal::Data> > idToData_;
            boost::shared_ptr<fwlite::HistoryGetterBase> historyGetter_;
            boost::shared_ptr<edm::EDProductGetter> getter_;
            mutable std::auto_ptr<TTreeCache> tcache_;
            mutable bool tcTrained_;
    };

}

#endif /*__CINT__ */

#endif
