#ifndef FWCore_ServiceRegistry_ServiceMaker_h
#define FWCore_ServiceRegistry_ServiceMaker_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceMaker
//
/**\class ServiceMaker ServiceMaker.h FWCore/ServiceRegistry/interface/ServiceMaker.h

 Description: Used to make an instance of a Service

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:00 EDT 2005
//

// user include files
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMakerBase.h"
#include "FWCore/ServiceRegistry/interface/ServicePluginFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceWrapper.h"
#include "FWCore/ServiceRegistry/interface/ServicesManager.h"

// system include files
#include <memory>
#include <typeinfo>

// forward declarations

namespace edm {
   class ActivityRegistry;
   class ParameterSet;

   namespace serviceregistry {

      template<typename T, typename TConcrete>
      struct MakerBase {
         typedef T interface_t;
         typedef TConcrete concrete_t;
      };

      template<typename T, typename TConcrete = T>
         struct AllArgsMaker : public MakerBase<T, TConcrete> {

         std::auto_ptr<T> make(ParameterSet const& iPS,
                               ActivityRegistry& iAR) const {
            return std::auto_ptr<T>(new TConcrete(iPS, iAR));
         }
      };

      template<typename T, typename TConcrete = T>
      struct ParameterSetMaker : public MakerBase<T, TConcrete> {
         std::auto_ptr<T> make(ParameterSet const& iPS,
                               ActivityRegistry& /* iAR */) const {
            return std::auto_ptr<T>(new TConcrete(iPS));
         }
      };

      template<typename T, typename TConcrete = T>
      struct NoArgsMaker : public MakerBase<T, TConcrete> {
         std::auto_ptr<T> make(ParameterSet const& /* iPS */,
                               ActivityRegistry& /* iAR */) const {
            return std::auto_ptr<T>(new TConcrete());
         }
      };

      template<typename T, typename TMaker = AllArgsMaker<T> >
      class ServiceMaker : public ServiceMakerBase {

public:
         ServiceMaker() {}
         //virtual ~ServiceMaker();

         // ---------- const member functions ---------------------
         virtual std::type_info const& serviceType() const { return typeid(T); }

         virtual bool make(ParameterSet const& iPS,
                           ActivityRegistry& iAR,
                           ServicesManager& oSM) const {
            TMaker maker;
            std::auto_ptr<T> pService(maker.make(iPS, iAR));
            boost::shared_ptr<ServiceWrapper<T> > ptr(new ServiceWrapper<T>(pService));
            return oSM.put(ptr);
         }

         virtual bool saveConfiguration() const {
            return ServiceMakerBase::testSaveConfiguration(static_cast<typename TMaker::concrete_t const*>(0));
         }

         // ---------- static member functions --------------------

         // ---------- member functions ---------------------------

private:
         ServiceMaker(ServiceMaker const&); // stop default

         ServiceMaker const& operator=(ServiceMaker const&); // stop default

         // ---------- member data --------------------------------

      };
   }
}

#define DEFINE_FWK_SERVICE(type) \
DEFINE_EDM_PLUGIN (edm::serviceregistry::ServicePluginFactory, edm::serviceregistry::ServiceMaker<type>, #type); \
DEFINE_DESC_FILLER_FOR_SERVICES(type, type)

#define DEFINE_FWK_SERVICE_MAKER(concrete, maker) \
typedef edm::serviceregistry::ServiceMaker<maker::interface_t, maker> concrete ## _ ## _t; \
DEFINE_EDM_PLUGIN (edm::serviceregistry::ServicePluginFactory, concrete ## _ ##  _t , #concrete); \
typedef maker::concrete_t concrete ## _ ## _ ## _t; \
DEFINE_DESC_FILLER_FOR_SERVICES(concrete, concrete ## _ ## _ ## _t)

#endif
