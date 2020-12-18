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

    template <typename T, typename TConcrete>
    struct MakerBase {
      typedef T interface_t;
      typedef TConcrete concrete_t;
    };

    template <typename T, typename TConcrete = T>
    struct AllArgsMaker : public MakerBase<T, TConcrete> {
      std::unique_ptr<T> make(ParameterSet const& iPS, ActivityRegistry& iAR) const {
        return std::make_unique<TConcrete>(iPS, iAR);
      }
    };

    template <typename T, typename TConcrete = T>
    struct ParameterSetMaker : public MakerBase<T, TConcrete> {
      std::unique_ptr<T> make(ParameterSet const& iPS, ActivityRegistry& /* iAR */) const {
        return std::make_unique<TConcrete>(iPS);
      }
    };

    template <typename T, typename TConcrete = T>
    struct NoArgsMaker : public MakerBase<T, TConcrete> {
      std::unique_ptr<T> make(ParameterSet const& /* iPS */, ActivityRegistry& /* iAR */) const {
        return std::make_unique<TConcrete>();
      }
    };

    template <typename T, typename TMaker = AllArgsMaker<T> >
    class ServiceMaker : public ServiceMakerBase {
    public:
      ServiceMaker() {}

      ServiceMaker(ServiceMaker const&) = delete;
      ServiceMaker const& operator=(ServiceMaker const&) = delete;

      // ---------- const member functions ---------------------
      std::type_info const& serviceType() const override { return typeid(T); }

      bool make(ParameterSet const& iPS, ActivityRegistry& iAR, ServicesManager& oSM) const override {
        TMaker maker;
        std::unique_ptr<T> pService(maker.make(iPS, iAR));
        auto ptr = std::make_shared<ServiceWrapper<T> >(std::move(pService));
        return oSM.put(ptr);
      }

      bool saveConfiguration() const override {
        return ServiceMakerBase::testSaveConfiguration(static_cast<typename TMaker::concrete_t const*>(nullptr));
      }

      bool processWideService() const override {
        return service::isProcessWideService(static_cast<typename TMaker::concrete_t const*>(nullptr));
      }
    };
  }  // namespace serviceregistry
}  // namespace edm

#define DEFINE_FWK_SERVICE(type)                                                                                  \
  DEFINE_EDM_PLUGIN(edm::serviceregistry::ServicePluginFactory, edm::serviceregistry::ServiceMaker<type>, #type); \
  DEFINE_DESC_FILLER_FOR_SERVICES(type, type)

#define DEFINE_FWK_SERVICE_MAKER(concrete, maker)                                            \
  typedef edm::serviceregistry::ServiceMaker<maker::interface_t, maker> concrete##_##_t;     \
  DEFINE_EDM_PLUGIN(edm::serviceregistry::ServicePluginFactory, concrete##_##_t, #concrete); \
  typedef maker::concrete_t concrete##_##_##_t;                                              \
  DEFINE_DESC_FILLER_FOR_SERVICES(concrete, concrete##_##_##_t)

#endif
