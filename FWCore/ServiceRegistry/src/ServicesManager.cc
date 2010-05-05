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
//

// system include files
#include <set>
#include <string>

// user include files
#include "FWCore/ServiceRegistry/interface/ServicesManager.h"
#include "FWCore/ServiceRegistry/interface/ServicePluginFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

using namespace edm::serviceregistry;
//
// constants, enums and typedefs
//

ServicesManager::MakerHolder::MakerHolder(boost::shared_ptr<ServiceMakerBase> iMaker,
                                          edm::ParameterSet& iPSet,
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
      wasAdded_ = maker_->make(*pset_, *registry_, oManager);
      if(wasAdded_ && maker_->saveConfiguration()) {
         pset_->addUntrackedParameter("@save_config",true);
      }
   }
   return wasAdded_;
}

//
// static data member definitions
//

//
// constructors and destructor
//
ServicesManager::ServicesManager(std::vector<edm::ParameterSet>& iConfiguration) :
type2Maker_(new Type2Maker)
{
   //First create the list of makers
   fillListOfMakers(iConfiguration);
   
   createServices();
}
ServicesManager::ServicesManager(ServiceToken iToken,
                                 ServiceLegacy iLegacy,
                                 std::vector<edm::ParameterSet>& iConfiguration):
  associatedManager_(iToken.manager_),
  type2Maker_(new Type2Maker)
{
   fillListOfMakers(iConfiguration);

   //find overlaps between services in iToken and iConfiguration
   typedef std::set< TypeIDBase> TypeSet;
   TypeSet configTypes;
   for(Type2Maker::iterator itType = type2Maker_->begin(), itTypeEnd = type2Maker_->end();
       itType != itTypeEnd;
       ++itType) {
      configTypes.insert(itType->first);
   }

   TypeSet tokenTypes;
   if(0 != associatedManager_.get()) {
      for(Type2Service::iterator itType = associatedManager_->type2Service_.begin(),
          itTypeEnd = associatedManager_->type2Service_.end();
          itType != itTypeEnd;
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
            for(IntersectionType::iterator itType = intersection.begin(), itTypeEnd = intersection.end();
                itType != itTypeEnd;
                ++itType) {
               Type2Maker::iterator itFound = type2Maker_->find(*itType);
               //HLT needs it such that even if a service isn't created we store is PSet if needed
               if(itFound->second.maker_->saveConfiguration()) {
                  itFound->second.pset_->addUntrackedParameter("@save_config",true);
               }
               type2Maker_->erase(itFound); 
            }
            break;
         case kConfigurationOverrides:
            //get all the services from Token
            type2Service_ = associatedManager_->type2Service_;
            
            //now remove the ones we do not want
            for(IntersectionType::iterator itType = intersection.begin(), itTypeEnd = intersection.end();
                itType != itTypeEnd;
                ++itType) {
               type2Service_.erase(type2Service_.find(*itType)); 
            }
            break;
      }
      //make sure our signals are propagated to our 'inherited' Services
      registry_.copySlotsFrom(associatedManager_->registry_); 
   }
   createServices();
}

// ServicesManager::ServicesManager(const ServicesManager& rhs)
// {
//    // do actual copying here;
// }

ServicesManager::~ServicesManager()
{
   // Force the Service destructors to execute in the reverse order of construction.
   // Note that services passed in by a token are not included in this loop and
   // do not get destroyed until the ServicesManager object that created them is destroyed
   // which occurs after the body of this destructor is executed (the correct order).
   // Services directly passed in by a put and not created in the constructor
   // may or not be detroyed in the desired order because this class does not control
   // their creation (as I'm writing this comment everything in a standard cmsRun
   // executable is destroyed in the desired order).
   for (std::vector<TypeIDBase>::const_reverse_iterator idIter = actualCreationOrder_.rbegin(),
                                                         idEnd = actualCreationOrder_.rend();
        idIter != idEnd;
        ++idIter) {

      Type2Service::iterator itService = type2Service_.find(*idIter);

      if (itService != type2Service_.end()) {

         // This will cause the Service's destruction if
         // there are no other shared pointers around
         itService->second.reset();
      }
   }
}

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
ServicesManager::fillListOfMakers(std::vector<edm::ParameterSet>& iConfiguration)
{
   for(std::vector<edm::ParameterSet>::iterator itParam = iConfiguration.begin(),
	itParamEnd = iConfiguration.end();
        itParam != itParamEnd;
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
      requestedCreationOrder_.push_back(TypeIDBase(base->serviceType()));
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
   
   for (std::vector<TypeIDBase>::const_iterator idIter = requestedCreationOrder_.begin(),
	                                         idEnd = requestedCreationOrder_.end();
        idIter != idEnd;
        ++idIter) {
     Type2Maker::iterator itMaker = type2Maker_->find(*idIter);

     // Check to make sure this maker is still there.  They are deleted
     // sometimes and that is OK.
     if (itMaker != type2Maker_->end()) {

       std::string serviceType = itMaker->second.pset_->getParameter<std::string>("@service_type");
       std::auto_ptr<edm::ParameterSetDescriptionFillerBase> filler(
         edm::ParameterSetDescriptionFillerPluginFactory::get()->create(serviceType));
       ConfigurationDescriptions descriptions(filler->baseType());

       try {
         filler->fill(descriptions);
       }
       catch (cms::Exception& iException) {
         edm::Exception toThrow(errors::Configuration, "Failed while filling ParameterSetDescriptions.");
         toThrow << "\nService plugin name is \"" << serviceType << "\"\n";
         toThrow.append(iException);
         throw toThrow;
       }

       try {
         descriptions.validate(*(itMaker->second.pset_), serviceType);
       }
       catch (cms::Exception& iException) {
         edm::Exception toThrow(errors::Configuration, "Failed validating service configuration.");
         toThrow << "\nService plugin name is \"" << serviceType << "\"\n";
         toThrow.append(iException);
         throw toThrow;
       }

       try {
         // This creates the service
         itMaker->second.add(*this);
       }
       catch(cms::Exception& iException) {
         edm::Exception toThrow(edm::errors::Configuration,"Error occurred while creating ");
         toThrow<<itMaker->second.pset_->getParameter<std::string>("@service_type")<<"\n";
         toThrow.append(iException);
         throw toThrow;
       }
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
