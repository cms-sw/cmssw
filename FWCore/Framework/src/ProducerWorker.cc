
/*----------------------------------------------------------------------
$Id: ProducerWorker.cc,v 1.15 2005/10/11 19:33:05 chrjones Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/ProducerWorker.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/interface/BranchDescription.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"


#include <string>
#include <sstream>
#include "boost/function.hpp"

using namespace std;

namespace edm
{
   static void addToRegistry(const EDProducer::TypeLabelList::const_iterator& iBegin,
                             const EDProducer::TypeLabelList::const_iterator& iEnd,
                             const ModuleDescription& iDesc,
                             ProductRegistry& iReg,
                             bool iIsListener=false){
      for(EDProducer::TypeLabelList::const_iterator p=iBegin; p!=iEnd; ++p) {
         
         
         BranchDescription pdesc(iDesc,
                                 p->typeID_.userClassName(),
                                 p->typeID_.friendlyClassName(), 
                                 p->productInstanceName_,
                                 p->productPtr_);
         iReg.addProduct(pdesc,iIsListener);
      }//for
   }
   namespace {
     class CallbackWrapper {
       public:  
        CallbackWrapper(boost::shared_ptr<EDProducer> iProd,
                        boost::function<void(const BranchDescription&)> iCallback,
                        ProductRegistry* iReg,
                        const ModuleDescription& iDesc):
        prod_(&(*iProd)), callback_(iCallback), reg_(iReg), mdesc_(iDesc),
        lastSize_(iProd->typeLabelList().size()) {}
        
        void operator()(const BranchDescription& iDesc){
           callback_(iDesc);
           addToRegistry();
        }
        
        void addToRegistry() {
           EDProducer::TypeLabelList const& plist = prod_->typeLabelList();
           
           if(lastSize_!=plist.size()){
              EDProducer::TypeLabelList::const_iterator pStart = plist.begin();
              advance(pStart,lastSize_);
              edm::addToRegistry(pStart,plist.end(),mdesc_,*reg_);
              lastSize_ = plist.size();
           }
        }

      private:
        EDProducer* prod_;
        boost::function<void(const BranchDescription&)> callback_;
        ProductRegistry* reg_;
        ModuleDescription mdesc_;
        unsigned int lastSize_;
         
     };
  }
   
  ProducerWorker::ProducerWorker(std::auto_ptr<EDProducer> ed,
				 const ModuleDescription& md,
				 const WorkerParams& wp):
    Worker(md,wp),
    producer_(ed) {
    
    if (producer_->typeLabelList().empty() && producer_->registrationCallback().empty()) {
      throw edm::Exception(errors::NoProductSpecified,"Producer")
        << "Module " << md.moduleName_
        << " did not specify that it produces a product.\n"
        << "The module constructor must call 'produces<T>(instanceName)'"
        << " for each product it produces.\nT is the product type.\n"
	<< "'instanceName' is an optional string used to distinguish" 
        << " multiple products of the same type.";
    }

    //If we have a callback, first tell the callback about all the entries already in the
    // product registry, then add any items this producer wants to add to the registry 
    // and only after that do we register the callback. This is done so the callback does not
    // get called for items registered by this producer (avoids circular reference problems)
    bool isListener = false;
    if(!(producer_->registrationCallback().empty())) {
       isListener=true;
       //NOTE: If implementation changes from a map, need to check that iterators are still valid
       // after an insert with the new container, else need to copy the container and iterate over the copy
       for(ProductRegistry::ProductList::const_iterator itEntry=wp.reg_->productList().begin();
           itEntry!=wp.reg_->productList().end(); ++itEntry){
          producer_->registrationCallback()(itEntry->second);
       }
    }
    EDProducer::TypeLabelList const& plist = producer_->typeLabelList();

    addToRegistry(plist.begin(),plist.end(),md,*(wp.reg_),isListener);
    if(!(producer_->registrationCallback().empty())) {
       Service<ConstProductRegistry> regService;
       regService->watchProductAdditions(CallbackWrapper(producer_,producer_->registrationCallback(),
                                                         wp.reg_,md));
    }
  }

  ProducerWorker::~ProducerWorker() {
  }

  bool ProducerWorker::implDoWork(EventPrincipal& ep, EventSetup const& c) {
    bool rc = false;

    Event e(ep,description());
    producer_->produce(e,c);
    e.commit_();
    rc=true;
    return rc;
  }

  void ProducerWorker::implBeginJob(EventSetup const& es) {
    producer_->beginJob(es);
  }

  void ProducerWorker::implEndJob() {
    producer_->endJob();
  }
  
  std::string ProducerWorker::workerType() const {
    return "EDProducer";
  }
  
}
