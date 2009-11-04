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
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "DataFormats/Provenance/interface/PassID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ComponentMaker.h"
#include "FWCore/Utilities/interface/EDMException.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupProvider;
      
template<typename T>
  class ComponentFactory
{

   public:
   ComponentFactory(): makers_() {}
   //~ComponentFactory();

   typedef  ComponentMakerBase<T> Maker;
   typedef std::map<std::string, boost::shared_ptr<Maker> > MakerMap;
   typedef typename T::base_type base_type;
      // ---------- const member functions ---------------------
   boost::shared_ptr<base_type> addTo(EventSetupProvider& iProvider,
                  edm::ParameterSet const& iConfiguration,
                  std::string const& iProcessName,
                  ReleaseVersion const& iVersion,
                  PassID const& iPass) const
      {
         std::string modtype = iConfiguration.template getParameter<std::string>("@module_type");
         //cerr << "Factory: module_type = " << modtype << endl;
         typename MakerMap::iterator it = makers_.find(modtype);
         
         if(it == makers_.end())
         {
            boost::shared_ptr<Maker> wm(edmplugin::PluginFactory<ComponentMakerBase<T>* ()>::get()->create(modtype));
            
            if(wm.get() == 0) {
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
               makers_.insert(std::make_pair<std::string,boost::shared_ptr<Maker> >(modtype,wm));
            
            if(ret.second == false) {
	      Exception::throwThis(errors::Configuration,"Maker Factory map insert failed");
            }
            
            it = ret.first;
         }
         
         try {
            return it->second->addTo(iProvider,iConfiguration,iProcessName,iVersion,iPass);
         } catch(cms::Exception& iException) {
            Exception toThrow(errors::Configuration,"Error occurred while creating ");
            toThrow<<modtype<<"\n";
            toThrow.append(iException);
            toThrow.raise();
         }
         return boost::shared_ptr<base_type>();
      }
   
      // ---------- static member functions --------------------
      static ComponentFactory<T>* get();

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
static edm::eventsetup::ComponentFactory<_type_> s_dummyfactory; \
namespace edm { namespace eventsetup { \
template<> edm::eventsetup::ComponentFactory<_type_>* edm::eventsetup::ComponentFactory<_type_>::get() \
{ return &s_dummyfactory; } \
  } } \
typedef int componentfactory_get_needs_semicolon

#endif
