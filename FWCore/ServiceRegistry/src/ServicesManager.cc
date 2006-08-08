// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServicesManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:19 EDT 2005
// $Id: ServicesManager.cc,v 1.7 2005/11/11 20:55:46 chrjones Exp $
//

// system include files
#include <set>

// user include files
#include "FWCore/ServiceRegistry/interface/ServicesManager.h"
#include "FWCore/ServiceRegistry/interface/ServicePluginFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm::serviceregistry;
//
// constants, enums and typedefs
//

ServicesManager::MakerHolder::MakerHolder(boost::shared_ptr<ServiceMakerBase> iMaker,
                                          const edm::ParameterSet& iPSet,
                                          edm::ActivityRegistry& iRegistry) :
maker_(iMaker),
pset_(&iPSet),
registry_(&iRegistry),
wasAdded_(false)
{}

bool 
ServicesManager::MakerHolder::add(ServicesManager& oManager) const
{
   if(!wasAdded_) {
      return wasAdded_ = maker_->make(*pset_, *registry_, oManager);
   }
   return wasAdded_;
}

//
// static data member definitions
//

//
// constructors and destructor
//
ServicesManager::ServicesManager(const std::vector<edm::ParameterSet>& iConfiguration) :
type2Maker_(new Type2Maker)
{
   //First create the list of makers
   fillListOfMakers(iConfiguration);
   
   createServices();
}
ServicesManager::ServicesManager(ServiceToken iToken,
                                 ServiceLegacy iLegacy,
                                 const std::vector<edm::ParameterSet>& iConfiguration):
type2Maker_(new Type2Maker),
associatedManager_(iToken.manager_)
{
   fillListOfMakers(iConfiguration);

   //find overlaps between services in iToken and iConfiguration
   typedef std::set< TypeIDBase> TypeSet;
   TypeSet configTypes;
   for(Type2Maker::iterator itType = type2Maker_->begin();
       itType != type2Maker_->end();
       ++itType) {
      configTypes.insert(itType->first);
   }

   TypeSet tokenTypes;
   if(0 != associatedManager_.get()) {
      for(Type2Service::iterator itType = associatedManager_->type2Service_.begin();
          itType != associatedManager_->type2Service_.end();
          ++itType) {
         tokenTypes.insert(itType->first);
      }
   
      typedef std::set<TypeIDBase> IntersectionType;
      IntersectionType intersection;
      std::set_intersection(configTypes.begin(), configTypes.end(),
                            tokenTypes.begin(), tokenTypes.end(),
                            inserter(intersection, intersection.end()));
      
      switch(iLegacy) {
         case kOverlapIsError :
            if(!intersection.empty()) {
               throw edm::Exception(errors::Configuration, "Service")
               <<"the Service "<<(*type2Maker_).find(*(intersection.begin()))->second.pset_->getParameter<std::string>("@service_type")
               <<" already has an instance of that type of Service";
            } else {
               //get all the services from Token
               type2Service_ = associatedManager_->type2Service_;
            }
            break;
         case kTokenOverrides :
            //get all the services from Token
            type2Service_ = associatedManager_->type2Service_;
            
            //remove from type2Maker the overlapping services so we never try to make them
            for(IntersectionType::iterator itType = intersection.begin();
                itType != intersection.end();
                ++itType) {
               type2Maker_->erase(type2Maker_->find(*itType)); 
            }
            break;
         case kConfigurationOverrides:
            //get all the services from Token
            type2Service_ = associatedManager_->type2Service_;
            
            //now remove the ones we do not want
            for(IntersectionType::iterator itType = intersection.begin();
                itType != intersection.end();
                ++itType) {
               type2Service_.erase(type2Service_.find(*itType)); 
            }
            break;
      }
   }
   createServices();

   //make sure our signals are propagated to our 'inherited' Services
   if(0 != associatedManager_.get()) {
      registry_.connect(associatedManager_->registry_);
   }
}

// ServicesManager::ServicesManager(const ServicesManager& rhs)
// {
//    // do actual copying here;
// }

//ServicesManager::~ServicesManager()
//{
//}
   
//
// assignment operators
//
// const ServicesManager& ServicesManager::operator=(const ServicesManager& rhs)
// {
//   //An exception safe implementation is
//   ServicesManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
ServicesManager::connect(ActivityRegistry& iOther)
{
   registry_.connect(iOther);
}   

void 
ServicesManager::connectTo(ActivityRegistry& iOther)
{
   iOther.connect(registry_);
}

void 
ServicesManager::copySlotsFrom(ActivityRegistry& iOther)
{
  registry_.copySlotsFrom(iOther);
}   

void 
ServicesManager::copySlotsTo(ActivityRegistry& iOther)
{
  iOther.copySlotsFrom(registry_);
}


void
ServicesManager::fillListOfMakers(const std::vector<edm::ParameterSet>& iConfiguration)
{
   for(std::vector<edm::ParameterSet>::const_iterator itParam = iConfiguration.begin();
        itParam != iConfiguration.end();
        ++itParam) {
      boost::shared_ptr<ServiceMakerBase> base(
                                               ServicePluginFactory::get()->create(itParam->getParameter<std::string>("@service_type")));
      if(0 == base.get()) {
         throw edm::Exception(edm::errors::Configuration, "Service")
         <<"could not find a service named "
         << itParam->getParameter<std::string>("@service_type")
         <<". Please check spelling.";
      }
      Type2Maker::iterator itFound = type2Maker_->find(TypeIDBase(base->serviceType()));
      if(itFound != type2Maker_->end()) {
         throw edm::Exception(edm::errors::Configuration,"Service") 
         <<" the service "<< itParam->getParameter<std::string>("@service_type") 
         <<" provides the same service as "
         << itFound->second.pset_->getParameter<std::string>("@service_type")
         <<"\n Please reconfigure job to only use one of these services.";
      }
      type2Maker_->insert(Type2Maker::value_type(TypeIDBase(base->serviceType()),
                                                  MakerHolder(base,
                                                              *itParam,
                                                              registry_)));
   }
   
}

namespace {
   struct NoOp {
      void operator()(ServicesManager*) {}
   };
}

void
ServicesManager::createServices()
{
   

   //create a shared_ptr of 'this' that will not delete us
   boost::shared_ptr<ServicesManager> shareThis(this, NoOp());
   
   ServiceToken token(shareThis);
   
   //Now make our services to ones obtained via ServiceRegistry
   // when this goes out of scope, it will revert back to the previous Service set
   ServiceRegistry::Operate operate(token);
   
   //Now, make each Service.  If a service depends on a service that has yet to be
   // created, that other service will automatically be made
   
   
   for(Type2Maker::iterator itMaker = type2Maker_->begin();
        itMaker != type2Maker_->end();
        ++itMaker) {
      try{
         itMaker->second.add(*this);
      }catch(cms::Exception& iException){
         edm::Exception toThrow(edm::errors::Configuration,"Error occured while creating ");
         toThrow<<itMaker->second.pset_->getParameter<std::string>("@service_type")<<"\n";
         toThrow.append(iException);
         throw toThrow;
      }
   }
   
   //No longer need the makers
   type2Maker_.reset();
   
}   
//
// const member functions
//

//
// static member functions
//
