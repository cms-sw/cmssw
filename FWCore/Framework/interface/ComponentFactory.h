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
// $Id: ComponentFactory.h,v 1.7 2005/07/20 03:00:36 jbk Exp $
//

// system include files
#include <string>
#include <map>
#include "boost/shared_ptr.hpp"

// user include files
#include "PluginManager/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ComponentMaker.h"
#include "FWCore/Utilities/interface/EDMException.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupProvider;
      
template< class T>
class ComponentFactory : public seal::PluginFactory< ComponentMakerBase<T>* ()>
{

   public:
   ComponentFactory() : seal::PluginFactory<ComponentMakerBase<T>* () >(
                                                                        T::name()) {}
   //~ComponentFactory();

   typedef  ComponentMakerBase<T> Maker;
   typedef std::map<std::string, boost::shared_ptr<Maker> > MakerMap;
      // ---------- const member functions ---------------------
      void addTo(EventSetupProvider& iProvider,
                  edm::ParameterSet const& iConfiguration,
                  std::string const& iProcessName,
                  unsigned long iVersion,
                  unsigned long iPass) const
      {
         using namespace std;
         string modtype = iConfiguration.template getParameter<string>("module_type");
         //cerr << "Factory: module_type = " << modtype << endl;
         typename MakerMap::iterator it = makers_.find(modtype);
         
         if(it == makers_.end())
         {
            boost::shared_ptr<Maker> wm(this->create(modtype));
            
            if(wm.get()==0) {
	      throw edm::Exception(errors::Configuration,(T::name()+ " failed to create a " + modtype).c_str());
            }
            
            //cerr << "Factory: created the worker" << endl;
            
            pair<typename MakerMap::iterator,bool> ret =
               makers_.insert(make_pair<string,boost::shared_ptr<Maker> >(modtype,wm));
            
            if(ret.second==false)
	      throw edm::Exception(errors::Configuration,"Maker Factory map insert failed");
            
            it = ret.first;
         }
         
         it->second->addTo(iProvider,iConfiguration,iProcessName,iVersion,iPass);
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
#define COMPONENTFACTORY_GET(_type_) static edm::eventsetup::ComponentFactory<_type_> s_dummyfactory; template<> edm::eventsetup::ComponentFactory<_type_>* edm::eventsetup::ComponentFactory<_type_>::get() { return &s_dummyfactory; }

#endif /* Framework_ComponentFactory_h */
