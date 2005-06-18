#ifndef EVENTSETUP_COMPONENTFACTORY_H
#define EVENTSETUP_COMPONENTFACTORY_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     ComponentFactory
// 
/**\class ComponentFactory ComponentFactory.h Core/CoreFramework/interface/ComponentFactory.h

 Description: Factory for building the Factories for the various 'plug-in' components needed for the EventSetup

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Wed May 25 15:21:05 EDT 2005
// $Id: ComponentFactory.h,v 1.3 2005/06/14 23:15:13 wmtan Exp $
//

// system include files
#include <string>
#include <map>
#include "boost/shared_ptr.hpp"

// user include files
#include "PluginManager/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/CoreFramework/interface/ComponentMaker.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupProvider;
      
template< class T>
class ComponentFactory : public seal::PluginFactory< ComponentMakerBase<T>* ()>
{

   public:
   ComponentFactory() : seal::PluginFactory<ComponentMakerBase<T>* () >(
                                                                        T::name() ) {}
   //~ComponentFactory();

   typedef  ComponentMakerBase<T> Maker;
   typedef std::map<std::string, boost::shared_ptr<Maker> > MakerMap;
      // ---------- const member functions ---------------------
      void addTo( EventSetupProvider& iProvider,
                  edm::ParameterSet const& iConfiguration,
                  std::string const& iProcessName,
                  unsigned long iVersion,
                  unsigned long iPass ) const
      {
         using namespace std;
         string modtype = iConfiguration.template getParameter<string>("module_type");
         //cerr << "Factory: module_type = " << modtype << endl;
         typename MakerMap::iterator it = makers_.find(modtype);
         
         if(it == makers_.end())
         {
            boost::shared_ptr<Maker> wm(this->create(modtype));
            
            if(wm.get()==0) {
               throw runtime_error((T::name()+ " failed to create a " + modtype).c_str());
            }
            
            //cerr << "Factory: created the worker" << endl;
            
            pair<typename MakerMap::iterator,bool> ret =
               makers_.insert(make_pair<string,boost::shared_ptr<Maker> >(modtype,wm));
            
            if(ret.second==false)
               throw runtime_error("Maker Factory map insert failed");
            
            it = ret.first;
         }
         
         it->second->addTo(iProvider,iConfiguration,iProcessName,iVersion,iPass);
      }
   
      // ---------- static member functions --------------------
      static ComponentFactory<T>* get();

      // ---------- member functions ---------------------------

   private:
      
      ComponentFactory( const ComponentFactory& ); // stop default

      const ComponentFactory& operator=( const ComponentFactory& ); // stop default

      // ---------- member data --------------------------------
      mutable MakerMap makers_;
};

   }
}
#define COMPONENTFACTORY_GET(_type_) static edm::eventsetup::ComponentFactory<_type_> s_dummyfactory; template<> edm::eventsetup::ComponentFactory<_type_>* edm::eventsetup::ComponentFactory<_type_>::get() { return &s_dummyfactory; }

#endif /* EVENTSETUP_COMPONENTFACTORY_H */
