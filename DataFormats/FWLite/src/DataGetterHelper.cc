// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     DataGetterHelper
//
// Implementation:
//     [Notes on implementation]
//
// Original Author: Eric Vaandering
//         Created:  Fri Jan 29 11:58:01 CST 2010
//

// system include files
#include <iostream>
#include "Reflex/Type.h"
#include "Reflex/Object.h"
#include "Reflex/Member.h"

// user include files
#include "DataFormats/FWLite/interface/DataGetterHelper.h"
#include "TFile.h"
#include "TTree.h"
#include "TTreeCache.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/WrapperHolder.h"

#include "FWCore/FWLite/interface/setRefStreamer.h"

#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "FWCore/Utilities/interface/EDMException.h"

#define TTCACHE_SIZE 20*1024*1024

namespace fwlite {
    //
    // constants, enums and typedefs
    //

    //
    // DataGetterHelper takes ownership of the TTreeCache, so make use of it by
    // the file exception safe
    //
    class withTCache {
    public:
        withTCache(TFile* file, TTreeCache* tc) : f_(file) { f_->SetCacheRead(tc); }
        ~withTCache() { f_->SetCacheRead(0); }
    private:
        TFile* f_;
    };

    //
    // static data member definitions
    //
    typedef std::map<internal::DataKey, boost::shared_ptr<internal::Data> > DataMap;
    // empty object used to signal that the branch requested was not found
    static internal::Data branchNotFound;

    //
    // constructors and destructor
    //
    DataGetterHelper::DataGetterHelper(TTree* tree,
                                       boost::shared_ptr<HistoryGetterBase> historyGetter,
                                       boost::shared_ptr<BranchMapReader> branchMap,
                                       boost::shared_ptr<edm::EDProductGetter> getter,
                                       bool useCache):
        branchMap_(branchMap),
        historyGetter_(historyGetter),
        getter_(getter),
        tcache_(0),
        tcTrained_(false)
    {
        if(0==tree) {
            throw cms::Exception("NoTree")<<"The TTree pointer passed to the constructor was null";
        }
        tree_ = tree;
        if (useCache) {
            tree_->SetCacheSize(TTCACHE_SIZE);
            TFile* iFile(branchMap_->getFile());
            tcache_.reset(dynamic_cast<TTreeCache*>(iFile->GetCacheRead()));
            iFile->SetCacheRead(0);
            //std::cout << "In const " << iFile << " " << tcache_ << " " << iFile->GetCacheRead() << std::endl;
        }
    }

    // DataGetterHelper::DataGetterHelper(const DataGetterHelper& rhs)
    // {
    //    // do actual copying here;
    // }

    DataGetterHelper::~DataGetterHelper() {    }

    //
    // assignment operators
    //
    // const DataGetterHelper& DataGetterHelper::operator=(const DataGetterHelper& rhs)
    // {
    //   //An exception safe implementation is
    //   DataGetterHelper temp(rhs);
    //   swap(rhs);
    //
    //   return *this;
    // }

    //
    // member functions
    //



    //
    // const member functions
    //

    //
    // static member functions
    //


    static
    TBranch* findBranch(TTree* iTree, std::string const& iMainLabels, std::string const& iProcess) {
        std::string branchName(iMainLabels);
        branchName+=iProcess;
        //branchName+=".obj";
        branchName+=".";
        return iTree->GetBranch(branchName.c_str());
    }

    void
    DataGetterHelper::getBranchData(edm::EDProductGetter* iGetter,
                    Long64_t index,
                    internal::Data& iData) const
    {
        GetterOperate op(iGetter);

        //WORK AROUND FOR ROOT!!
        //Create a new instance so that we can clear any cache the object uses
        //this slows the code down
        Reflex::Object obj = iData.obj_;
        iData.obj_ = iData.obj_.TypeOf().Construct();
        iData.pObj_ = iData.obj_.Address();
        iData.branch_->SetAddress(&(iData.pObj_));
        //If a REF to this was requested in the past, we might as well do the work now
        if(0!=iData.pProd_) {
            iData.pProd_ = iData.obj_.Address();
        }
        obj.Destruct();
        //END OF WORK AROUND

        if (0 == tcache_.get()) {
            iData.branch_->GetEntry(index);
        } else {
            if (!tcTrained_) {
                tcache_->SetLearnEntries(100);
                tcache_->SetEntryRange(0, tree_->GetEntries());
                tcTrained_ = true;
            }
            withTCache tcguard(branchMap_->getFile(), tcache_.get());
            tree_->LoadTree(index);
            iData.branch_->GetEntry(index);
       }
       iData.lastProduct_=index;
    }


    internal::Data&
    DataGetterHelper::getBranchDataFor(std::type_info const& iInfo,
                    char const* iModuleLabel,
                    char const* iProductInstanceLabel,
                    char const* iProcessLabel) const
    {
        edm::TypeID type(iInfo);
        internal::DataKey key(type, iModuleLabel, iProductInstanceLabel, iProcessLabel);

        boost::shared_ptr<internal::Data> theData;
        DataMap::iterator itFind = data_.find(key);
        if(itFind == data_.end()) {
            //see if such a branch actually exists
            std::string const sep("_");
            //CHANGE: If this fails, need to lookup the the friendly name which was used to write the file
            std::string name(type.friendlyClassName());
            name +=sep+std::string(key.module());
            name +=sep+std::string(key.product())+sep;

            //if we have to lookup the process label, remember it and register the product again
            std::string foundProcessLabel;
            TBranch* branch = 0;

            if (0==iProcessLabel || iProcessLabel==key.kEmpty() ||
                strlen(iProcessLabel)==0) {
                std::string const* lastLabel=0;
                //have to search in reverse order since newest are on the bottom
                const edm::ProcessHistory& h = DataGetterHelper::history();
                for (edm::ProcessHistory::const_reverse_iterator iproc = h.rbegin(), eproc = h.rend();
                    iproc != eproc;
                    ++iproc) {

                    lastLabel = &(iproc->processName());
                    branch=findBranch(tree_,name,iproc->processName());
                    if(0!=branch) {
                    break;
                    }
                }
                if(0==branch) {
                    return branchNotFound;
                }
                //do we already have this one?
                if(0!=lastLabel) {
                    internal::DataKey fullKey(type,iModuleLabel,iProductInstanceLabel,lastLabel->c_str());
                    itFind = data_.find(fullKey);
                    if(itFind != data_.end()) {
                        //remember the data we've found
                        theData = itFind->second;
                    } else {
                        //only set this if we don't already have it since it this string is not empty we re-register
                        foundProcessLabel = *lastLabel;
                    }
                }
            } else {
                //we have all the pieces
                branch = findBranch(tree_,name,key.process());
                if(0==branch){
                    return branchNotFound;
                }
            }

            //cache the info
            size_t moduleLabelLen = strlen(iModuleLabel)+1; 
            char* newModule = new char[moduleLabelLen];
            std::strncpy(newModule,iModuleLabel,moduleLabelLen);
            labels_.push_back(newModule);

            char* newProduct = const_cast<char*>(key.product());
            if(newProduct[0] != 0) {
                size_t newProductLen = strlen(newProduct)+1; 
                newProduct = new char[newProductLen];
                std::strncpy(newProduct,key.product(),newProductLen);
                labels_.push_back(newProduct);
            }
            char* newProcess = const_cast<char*>(key.process());
            if(newProcess[0]!=0) {
                size_t newProcessLen = strlen(newProcess)+1; 
                newProcess = new char[newProcessLen];
                std::strncpy(newProcess,key.process(),newProcessLen);
                labels_.push_back(newProcess);
            }
            internal::DataKey newKey(edm::TypeID(iInfo),newModule,newProduct,newProcess);

            if(0 == theData.get()) {
                //We do not already have this data as another key

                //Use Reflex to create an instance of the object to be used as a buffer
                Reflex::Type rType = Reflex::Type::ByTypeInfo(iInfo);
                if(rType == Reflex::Type()) {
                    throw cms::Exception("UnknownType")<<"No Reflex dictionary exists for type "<<iInfo.name();
                }
                Reflex::Object obj = rType.Construct();

                if(obj.Address() == 0) {
                    throw cms::Exception("ConstructionFailed")<<"failed to construct an instance of "<<rType.Name();
                }
                boost::shared_ptr<internal::Data> newData(new internal::Data());
                newData->branch_ = branch;
                newData->obj_ = obj;
                newData->lastProduct_=-1;
                newData->pObj_ = obj.Address();
                newData->pProd_ = 0;
                branch->SetAddress(&(newData->pObj_));
                newData->interface_ = 0;
                Reflex::Member getTheInterface = rType.FunctionMemberByName(std::string("getInterface"));
                getTheInterface.Invoke(newData->interface_);
                theData = newData;
            }
            itFind = data_.insert(std::make_pair(newKey, theData)).first;

            if(foundProcessLabel.size()) {
                //also remember it with the process label
                newProcess = new char[foundProcessLabel.size()+1];
                std::strcpy(newProcess,foundProcessLabel.c_str());
                labels_.push_back(newProcess);
                internal::DataKey newKey(edm::TypeID(iInfo),newModule,newProduct,newProcess);

                data_.insert(std::make_pair(newKey,theData));
            }
        }
        return *(itFind->second);
    }

    std::string const
    DataGetterHelper::getBranchNameFor(std::type_info const& iInfo,
                    char const* iModuleLabel,
                    char const* iProductInstanceLabel,
                    char const* iProcessLabel) const
    {
        internal::Data& theData =
            DataGetterHelper::getBranchDataFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);

        if (0 != theData.branch_) {
            return std::string(theData.branch_->GetName());
        }
        return std::string("");
    }

    bool
    DataGetterHelper::getByLabel(std::type_info const& iInfo,
                    char const* iModuleLabel,
                    char const* iProductInstanceLabel,
                    char const* iProcessLabel,
                    void* oData, Long_t index) const
    {
        // Maintain atEnd() check in parent classes
        void** pOData = reinterpret_cast<void**>(oData);
        *pOData = 0;

        internal::Data& theData =
            DataGetterHelper::getBranchDataFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);

        if (0 != theData.branch_) {
            if(index != theData.lastProduct_) {
                //haven't gotten the data for this event
                getBranchData(getter_.get(), index, theData);
            }
            *pOData = theData.obj_.Address();
        }

        if (0 == *pOData) return false;
        else return true;
    }

    bool
    DataGetterHelper::getByLabel(std::type_info const& iInfo,
                    char const* iModuleLabel,
                    char const* iProductInstanceLabel,
                    char const* iProcessLabel,
                    edm::WrapperHolder& holder, Long_t index) const {

        // Maintain atEnd() check in parent classes

        internal::Data& theData =
            DataGetterHelper::getBranchDataFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);

        if(0 != theData.branch_) {
            if(index != theData.lastProduct_) {
                //haven't gotten the data for this event
                getBranchData(getter_.get(), index, theData);
            }
        }

        holder = edm::WrapperHolder(theData.obj_.Address(), theData.interface_);
        return holder.isValid();
    }

    edm::WrapperHolder
    DataGetterHelper::getByProductID(edm::ProductID const& iID, Long_t index) const
    {
        typedef std::pair<edm::ProductID,edm::BranchListIndexes> IDPair;
        IDPair theID = std::make_pair(iID, branchMap_->branchListIndexes());
        std::map<IDPair,boost::shared_ptr<internal::Data> >::const_iterator itFound = idToData_.find(theID);
        if(itFound == idToData_.end()) {
            edm::BranchDescription const& bDesc = branchMap_->productToBranch(iID);

            if (!bDesc.branchID().isValid()) {
                return edm::WrapperHolder();
            }

            //Calculate the key from the branch description
            Reflex::Type type(Reflex::Type::ByName(edm::wrappedClassName(bDesc.fullClassName())));
            assert(Reflex::Type() != type) ;

            //Only the product instance label may be empty
            char const* pIL = bDesc.productInstanceName().c_str();
            if(pIL[0] == 0) {
                pIL = 0;
            }
            internal::DataKey k(edm::TypeID(type.TypeInfo()),
                                bDesc.moduleLabel().c_str(),
                                pIL,
                                bDesc.processName().c_str());

            //has this already been gotten?
            KeyToDataMap::iterator itData = data_.find(k);
            if(data_.end() == itData) {
                //ask for the data
                edm::WrapperHolder holder;
                getByLabel(type.TypeInfo(),
                            k.module(),
                            k.product(),
                            k.process(),
                            holder, index);
                if(!holder.isValid()) {
                    return holder;
                }
                itData = data_.find(k);
                assert(itData != data_.end());
                assert(holder.wrapper() == itData->second->obj_.Address());
            }
            itFound = idToData_.insert(std::make_pair(theID,itData->second)).first;
        }
        if(index != itFound->second->lastProduct_) {
            //haven't gotten the data for this event
            getBranchData(getter_.get(), index, *(itFound->second));
        }
        if(0==itFound->second->pProd_) {
            itFound->second->pProd_ = itFound->second->obj_.Address();

            if(0==itFound->second->pProd_) {
              return edm::WrapperHolder();
            }
        }
        //return itFound->second->pProd_;
        return edm::WrapperHolder(itFound->second->pProd_, itFound->second->interface_);
    }

    const edm::ProcessHistory& DataGetterHelper::history() const {
        return historyGetter_->history();
    }


    //
    // static member functions
    //
    void
    DataGetterHelper::throwProductNotFoundException(std::type_info const& iType, char const* iModule, char const* iProduct, char const* iProcess)
    {
        edm::TypeID type(iType);
        throw edm::Exception(edm::errors::ProductNotFound)<<"A branch was found for \n  type ='"<<type.className()<<"'\n  module='"<<iModule
            <<"'\n  productInstance='"<<((0!=iProduct)?iProduct:"")<<"'\n  process='"<<((0!=iProcess)?iProcess:"")<<"'\n"
            "but no data is available for this Lumi";
    }

}
