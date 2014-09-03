#ifndef ServiceRegistry_ServicesManager_h
#define ServiceRegistry_ServicesManager_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServicesManager
//
/**\class ServicesManager ServicesManager.h FWCore/ServiceRegistry/interface/ServicesManager.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:01 EDT 2005
//

// user include files
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceMakerBase.h"
#include "FWCore/ServiceRegistry/interface/ServiceWrapper.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeIDBase.h"

// system include files
#include <memory>

#include <cassert>
#include <vector>

// forward declarations
namespace edm {
   class ParameterSet;
   class ServiceToken;

   namespace serviceregistry {

      class ServicesManager {
public:
         struct MakerHolder {
            MakerHolder(std::shared_ptr<ServiceMakerBase> iMaker,
                        ParameterSet& iPSet,
                        ActivityRegistry&) ;
            bool add(ServicesManager&) const;

            std::shared_ptr<ServiceMakerBase> maker_;
            ParameterSet* pset_;
            ActivityRegistry* registry_;
            mutable bool wasAdded_;
         };
         typedef std::map<TypeIDBase, std::shared_ptr<ServiceWrapperBase> > Type2Service;
         typedef std::map<TypeIDBase, MakerHolder> Type2Maker;

         ServicesManager(std::vector<ParameterSet>& iConfiguration);

         /** Takes the services described by iToken and places them into the manager.
             Conflicts over Services provided by both the iToken and iConfiguration
             are resolved based on the value of iLegacy
         */
         ServicesManager(ServiceToken iToken,
                         ServiceLegacy iLegacy,
                         std::vector<ParameterSet>& iConfiguration,
                         bool associate = true);

         ~ServicesManager();

         // ---------- const member functions ---------------------
         template<typename T>
         T& get() const {
            Type2Service::const_iterator itFound = type2Service_.find(TypeIDBase(typeid(T)));
            Type2Maker::const_iterator itFoundMaker ;
            if(itFound == type2Service_.end()) {
               //do on demand building of the service
               if(0 == type2Maker_.get() ||
                   type2Maker_->end() == (itFoundMaker = type2Maker_->find(TypeIDBase(typeid(T))))) {
                      Exception::throwThis(errors::NotFound,
                        "Service Request unable to find requested service with compiler type name '",
                        typeid(T).name(),
                        "'.\n");
               } else {
                  itFoundMaker->second.add(const_cast<ServicesManager&>(*this));
                  itFound = type2Service_.find(TypeIDBase(typeid(T)));
                  //the 'add()' should have put the service into the list
                  assert(itFound != type2Service_.end());
               }
            }
            //convert it to its actual type
            std::shared_ptr<ServiceWrapper<T> > ptr(std::dynamic_pointer_cast<ServiceWrapper<T> >(itFound->second));
            assert(0 != ptr.get());
            return ptr->get();
         }

         ///returns true of the particular service is accessible
         template<typename T>
         bool isAvailable() const {
            Type2Service::const_iterator itFound = type2Service_.find(TypeIDBase(typeid(T)));
            Type2Maker::const_iterator itFoundMaker ;
            if(itFound == type2Service_.end()) {
               //do on demand building of the service
               if(0 == type2Maker_.get() ||
                   type2Maker_->end() == (itFoundMaker = type2Maker_->find(TypeIDBase(typeid(T))))) {
                  return false;
               } else {
                  //Actually create the service in order to 'flush out' any
                  // configuration errors for the service
                  itFoundMaker->second.add(const_cast<ServicesManager&>(*this));
                  itFound = type2Service_.find(TypeIDBase(typeid(T)));
                  //the 'add()' should have put the service into the list
                  assert(itFound != type2Service_.end());
               }
            }
            return true;
         }

         // ---------- static member functions --------------------

         // ---------- member functions ---------------------------
         ///returns false if put fails because a service of this type already exists
         template<typename T>
         bool put(std::shared_ptr<ServiceWrapper<T> > iPtr) {
            Type2Service::const_iterator itFound = type2Service_.find(TypeIDBase(typeid(T)));
            if(itFound != type2Service_.end()) {
               return false;
            }
            type2Service_[ TypeIDBase(typeid(T)) ] = iPtr;
            actualCreationOrder_.push_back(TypeIDBase(typeid(T)));
            return true;
         }

         ///causes our ActivityRegistry's signals to be forwarded to iOther
         void connect(ActivityRegistry& iOther);

         ///causes iOther's signals to be forward to us
         void connectTo(ActivityRegistry& iOther);

         ///copy our Service's slots to the argument's signals
         void copySlotsTo(ActivityRegistry&);
         ///the copy the argument's slots to the our signals
         void copySlotsFrom(ActivityRegistry&);

private:
         ServicesManager(ServicesManager const&); // stop default

         ServicesManager const& operator=(ServicesManager const&); // stop default

         void fillListOfMakers(std::vector<ParameterSet>&);
         void createServices();

         // ---------- member data --------------------------------
         //hold onto the Manager passed in from the ServiceToken so that
         // the ActivityRegistry of that Manager does not go out of scope
         // This must be first to get the Service destructors called in
         // the correct order.
         std::shared_ptr<ServicesManager> associatedManager_;

         ActivityRegistry registry_;
         Type2Service type2Service_;
         std::auto_ptr<Type2Maker> type2Maker_;
         std::vector<TypeIDBase> requestedCreationOrder_;
         std::vector<TypeIDBase> actualCreationOrder_;
      };
   }
}

#endif
