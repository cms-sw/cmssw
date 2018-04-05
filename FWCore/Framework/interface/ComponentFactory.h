#ifndef Framework_ComponentFactory_h
#define Framework_ComponentFactory_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ComponentFactory
// 
/**\class ComponentFactory ComponentFactory.h FWCore/Framework/interface/ComponentFactory.h

 Description: Factory for building the Factories for the various 'plug-in' components needed for the EventSetup

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Wed May 25 15:21:05 EDT 2005
//

// system include files
#include <string>
#include <map>
#include <memory>
#include <exception>

// user include files
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ComponentMaker.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupProvider;
      class EventSetupsController;
      
template<typename T>
  class ComponentFactory
{

   public:
   ComponentFactory(): makers_() {}
   //~ComponentFactory();

   typedef  ComponentMakerBase<T> Maker;
   typedef std::map<std::string, std::shared_ptr<Maker> > MakerMap;
   typedef typename T::base_type base_type;
      // ---------- const member functions ---------------------
   std::shared_ptr<base_type> addTo(EventSetupsController& esController,
                                      EventSetupProvider& iProvider,
                                      edm::ParameterSet const& iConfiguration,
                                      bool replaceExisting = false) const
      {
         std::string modtype = iConfiguration.template getParameter<std::string>("@module_type");
         //cerr << "Factory: module_type = " << modtype << endl;
         typename MakerMap::iterator it = makers_.find(modtype);
         
         if(it == makers_.end())
         {
            std::shared_ptr<Maker> wm(edmplugin::PluginFactory<ComponentMakerBase<T>* ()>::get()->create(modtype));
            
            if(wm.get() == nullptr) {
	      Exception::throwThis(errors::Configuration,
	      "UnknownModule",
	       T::name().c_str(),
              " of type ",
              modtype.c_str(),
              " has not been registered.\n"
              "Perhaps your module type is misspelled or is not a "
              "framework plugin.\n"
              "Try running EdmPluginDump to obtain a list of "
              "available Plugins.");            
            }
            
            //cerr << "Factory: created the worker" << endl;
            
            std::pair<typename MakerMap::iterator,bool> ret =
               makers_.insert(std::pair<std::string,std::shared_ptr<Maker> >(modtype,wm));
            
            if(ret.second == false) {
	      Exception::throwThis(errors::Configuration,"Maker Factory map insert failed");
            }
            
            it = ret.first;
         }
         
         try {
           return convertException::wrap([&]() -> std::shared_ptr<base_type> {
             return it->second->addTo(esController, iProvider, iConfiguration, replaceExisting);
           });
         }
         catch(cms::Exception & iException) {
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
      
      ComponentFactory(const ComponentFactory&); // stop default

      const ComponentFactory& operator=(const ComponentFactory&); // stop default

      // ---------- member data --------------------------------
      mutable MakerMap makers_;
};

   }
}
#define COMPONENTFACTORY_GET(_type_) \
EDM_REGISTER_PLUGINFACTORY(edmplugin::PluginFactory<edm::eventsetup::ComponentMakerBase<_type_>* ()>,_type_::name()); \
static edm::eventsetup::ComponentFactory<_type_> const s_dummyfactory; \
namespace edm { namespace eventsetup { \
template<> edm::eventsetup::ComponentFactory<_type_> const* edm::eventsetup::ComponentFactory<_type_>::get() \
{ return &s_dummyfactory; } \
  } } \
typedef int componentfactory_get_needs_semicolon

#endif
