// -*- C++ -*-
#ifndef Framework_ComponentFactory_h
#define Framework_ComponentFactory_h
//
// Package:     Framework
// Class  :     ComponentFactory
//
/**\class edm::eventsetup::ComponentFactory

 Description: Factory for building the Factories for the various 'plug-in' components needed for the EventSetup

 Usage:
    NOTE: it is not safe to call this class across threads, even if only calling const member functions.

*/
//
// Author:      Chris Jones
// Created:     Wed May 25 15:21:05 EDT 2005
//

// system include files
#include <string>
#include <map>
#include <memory>
#include <sstream>

// user include files
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ComponentMaker.h"
#include "FWCore/Framework/interface/resolveMaker.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

// forward declarations
namespace edm {
  class ModuleTypeResolverMaker;

  namespace eventsetup {
    class EventSetupProvider;

    template <typename T>
    class ComponentFactory {
    public:
      ComponentFactory() : makers_() {}
      ComponentFactory(const ComponentFactory&) = delete;
      ComponentFactory(ComponentFactory&&) = delete;
      ComponentFactory& operator=(ComponentFactory&&) = delete;
      const ComponentFactory& operator=(const ComponentFactory&) = delete;

      typedef ComponentMakerBase<T> Maker;
      typedef std::map<std::string, std::shared_ptr<Maker const>> MakerMap;
      typedef typename T::base_type base_type;
      // ---------- const member functions ---------------------
      std::shared_ptr<base_type> addTo(EventSetupProvider& iProvider,
                                       edm::ParameterSet& iConfiguration,
                                       ModuleTypeResolverMaker const* resolverMaker) const {
        std::string modtype = iConfiguration.template getParameter<std::string>("@module_type");
        //cerr << "Factory: module_type = " << modtype << endl;
        typename MakerMap::iterator it = makers_.find(modtype);
        Maker const* maker = nullptr;

        if (it == makers_.end()) {
          maker = detail::resolveMaker<edmplugin::PluginFactory<ComponentMakerBase<T>*()>>(
              modtype, resolverMaker, iConfiguration, makers_);

          if (not maker) {
            Exception::throwThis(errors::Configuration, "Maker Factory map insert failed");
          }
        } else {
          maker = it->second.get();
        }

        try {
          return convertException::wrap(
              [&]() -> std::shared_ptr<base_type> { return maker->addTo(iProvider, iConfiguration); });
        } catch (cms::Exception& iException) {
          std::string edmtype = iConfiguration.template getParameter<std::string>("@module_edm_type");
          std::string label = iConfiguration.template getParameter<std::string>("@module_label");
          std::ostringstream ost;
          ost << "Constructing " << edmtype << ": class=" << modtype << " label='" << label << "'";
          iException.addContext(ost.str());
          throw;
        }
        return std::shared_ptr<base_type>();
      }

      // ---------- static member functions --------------------
      static ComponentFactory<T> const* get();

      // ---------- member functions ---------------------------

    private:
      // ---------- member data --------------------------------
      //Creating a new component is not done across threads
      CMS_SA_ALLOW mutable MakerMap makers_;
    };

  }  // namespace eventsetup
}  // namespace edm
#define COMPONENTFACTORY_GET(_type_)                                                                      \
  EDM_REGISTER_PLUGINFACTORY(edmplugin::PluginFactory<edm::eventsetup::ComponentMakerBase<_type_>*()>,    \
                             _type_::name());                                                             \
  static edm::eventsetup::ComponentFactory<_type_> const s_dummyfactory;                                  \
  namespace edm {                                                                                         \
    namespace eventsetup {                                                                                \
      template <>                                                                                         \
      edm::eventsetup::ComponentFactory<_type_> const* edm::eventsetup::ComponentFactory<_type_>::get() { \
        return &s_dummyfactory;                                                                           \
      }                                                                                                   \
    }                                                                                                     \
  }                                                                                                       \
  typedef int componentfactory_get_needs_semicolon

#endif
