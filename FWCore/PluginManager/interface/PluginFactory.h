#ifndef FWCore_PluginManager_PluginFactory_h
#define FWCore_PluginManager_PluginFactory_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginFactory
//
/**\class PluginFactory PluginFactory.h FWCore/PluginManager/interface/PluginFactory.h

 Description: Public interface for loading a plugin

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Apr  5 12:10:23 EDT 2007
//

// system include files
#include <map>
#include <memory>
#include <vector>

// user include files
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Utilities/interface/concatenate.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
// forward declarations

namespace edmplugin {
  template <class T>
  class PluginFactory;
  class DummyFriend;

  template <typename R, typename... Args>
  class PluginFactory<R*(Args...)> : public PluginFactoryBase {
    friend class DummyFriend;

  public:
    using TemplateArgType = R*(Args...);
    using CreatedType = R;

    PluginFactory(const PluginFactory&) = delete;                   // stop default
    const PluginFactory& operator=(const PluginFactory&) = delete;  // stop default

    struct PMakerBase {
      virtual std::unique_ptr<R> create(Args...) const = 0;
      virtual ~PMakerBase() {}
    };
    template <class TPlug>
    struct PMaker : public PMakerBase {
      PMaker(const std::string& iName) { PluginFactory<R*(Args...)>::get()->registerPMaker(this, iName); }
      std::unique_ptr<R> create(Args... args) const override {
        return std::make_unique<TPlug>(std::forward<Args>(args)...);
      }
    };

    // ---------- const member functions ---------------------
    const std::string& category() const override;

    std::unique_ptr<R> create(const std::string& iName, Args... args) const {
      return reinterpret_cast<PMakerBase*>(PluginFactoryBase::findPMaker(iName))->create(std::forward<Args>(args)...);
    }

    ///like above but returns 0 if iName is unknown
    std::unique_ptr<R> tryToCreate(const std::string& iName, Args... args) const {
      auto found = PluginFactoryBase::tryToFindPMaker(iName);
      if (found == nullptr) {
        return nullptr;
      }
      return reinterpret_cast<PMakerBase*>(found)->create(args...);
    }
    // ---------- static member functions --------------------

    static PluginFactory<R*(Args...)>* get();
    // ---------- member functions ---------------------------
    void registerPMaker(PMakerBase* iPMaker, const std::string& iName) {
      PluginFactoryBase::registerPMaker(iPMaker, iName);
    }

  private:
    PluginFactory() { finishedConstruction(); }
  };
}  // namespace edmplugin
#define EDM_REGISTER_PLUGINFACTORY(_factory_, _category_)                                                               \
  namespace edmplugin {                                                                                                 \
    template <>                                                                                                         \
    edmplugin::PluginFactory<_factory_::TemplateArgType>* edmplugin::PluginFactory<_factory_::TemplateArgType>::get() { \
      CMS_THREAD_SAFE static edmplugin::PluginFactory<_factory_::TemplateArgType> s_instance;                           \
      return &s_instance;                                                                                               \
    }                                                                                                                   \
    template <>                                                                                                         \
    const std::string& edmplugin::PluginFactory<_factory_::TemplateArgType>::category() const {                         \
      static const std::string s_cat(_category_);                                                                       \
      return s_cat;                                                                                                     \
    }                                                                                                                   \
  }                                                                                                                     \
  enum { EDM_CONCATENATE(dummy_edm_register_pluginfactory_, __LINE__) }

#define EDM_REGISTER_PLUGINFACTORY2(_factory_, _category_)                                                              \
  namespace edmplugin {                                                                                                 \
    template <>                                                                                                         \
    edmplugin::PluginFactory<_factory_::TemplateArgType>* edmplugin::PluginFactory<_factory_::TemplateArgType>::get() { \
      CMS_THREAD_SAFE static edmplugin::PluginFactory<_factory_::TemplateArgType> s_instance;                           \
      return &s_instance;                                                                                               \
    }                                                                                                                   \
    template <>                                                                                                         \
    const std::string& edmplugin::PluginFactory<_factory_::TemplateArgType>::category() const {                         \
      static const std::string s_cat(_category_);                                                                       \
      return s_cat;                                                                                                     \
    }                                                                                                                   \
  }                                                                                                                     \
  enum { EDM_CONCATENATE(dummy_edm_register_pluginfactory_2_, __LINE__) }

#endif

#define EDM_PLUGIN_SYM(x, y) EDM_PLUGIN_SYM2(x, y)
#define EDM_PLUGIN_SYM2(x, y) x##y

#define DEFINE_EDM_PLUGIN(factory, type, name) \
  static const factory::PMaker<type> EDM_PLUGIN_SYM(s_maker, __LINE__)(name)

#define DEFINE_EDM_PLUGIN2(factory, type, name) \
  static const factory::PMaker<type> EDM_PLUGIN_SYM(s_maker2, __LINE__)(name)
