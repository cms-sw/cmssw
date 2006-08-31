/*----------------------------------------------------------------------
  
$Id: ProducerBase.cc,v 1.1 2006/04/20 22:34:05 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Actions.h"
#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/ProductRegistry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"

#include <string>
#include <sstream>
#include "boost/function.hpp"

using namespace std;

namespace edm {
  ProducerBase::ProducerBase() : ProductRegistryHelper(), callWhenNewProductsRegistered_() {}
  ProducerBase::~ProducerBase() { }

   boost::function<void(const BranchDescription&)> ProducerBase::registrationCallback() const {
      return callWhenNewProductsRegistered_;
   }


   namespace {
     class CallbackWrapper {
       public:  
        CallbackWrapper(boost::shared_ptr<ProducerBase> iProd,
                        boost::function<void(const BranchDescription&)> iCallback,
                        ProductRegistry* iReg,
                        const ModuleDescription& iDesc):
        prod_(&(*iProd)), callback_(iCallback), reg_(iReg), mdesc_(iDesc),
        lastSize_(iProd->typeLabelList().size()) {}
        
        void operator()(const BranchDescription& iDesc) {
           callback_(iDesc);
           addToRegistry();
        }
        
        void addToRegistry() {
           ProducerBase::TypeLabelList const& plist = prod_->typeLabelList();
           
           if(lastSize_!=plist.size()){
              ProducerBase::TypeLabelList::const_iterator pStart = plist.begin();
              advance(pStart, lastSize_);
              ProductRegistryHelper::addToRegistry(pStart, plist.end() ,mdesc_, *reg_);
              lastSize_ = plist.size();
           }
        }

      private:
        ProducerBase* prod_;
        boost::function<void(const BranchDescription&)> callback_;
        ProductRegistry* reg_;
        ModuleDescription mdesc_;
        unsigned int lastSize_;
         
     };
  }


  void ProducerBase::registerProducts(boost::shared_ptr<ProducerBase> product,
				ProductRegistry* iReg,
				ModuleDescription const& md,
				bool throwIfNoProducts)
  {
    if (typeLabelList().empty() && registrationCallback().empty()) {
      if (throwIfNoProducts) {
        throw edm::Exception(errors::NoProductSpecified,"Producer")
		<< "Module " << md.moduleName_
		<< " did not specify that it produces a product.\n"
		<< "The module constructor must call 'produces<T>(instanceName)'"
		<< " for each product it produces.\nT is the product type.\n"
		<< "'instanceName' is an optional string used to distinguish"
		<< " multiple products of the same type.";
      } else {
        return;
      }
    }
    //If we have a callback, first tell the callback about all the entries already in the
    // product registry, then add any items this producer wants to add to the registry 
    // and only after that do we register the callback. This is done so the callback does not
    // get called for items registered by this producer (avoids circular reference problems)
    bool isListener = false;
    if(!(registrationCallback().empty())) {
       isListener=true;
       iReg->callForEachBranch(registrationCallback());
    }
    TypeLabelList const& plist = typeLabelList();

    ProductRegistryHelper::addToRegistry(plist.begin(), plist.end(), md, *(iReg), isListener);
    if(!(registrationCallback().empty())) {
       Service<ConstProductRegistry> regService;
       regService->watchProductAdditions(CallbackWrapper(product, registrationCallback(), iReg, md));
    }
  }
}
