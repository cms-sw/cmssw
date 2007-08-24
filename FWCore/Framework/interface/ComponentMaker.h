#ifndef Framework_ComponentMaker_h
#define Framework_ComponentMaker_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ComponentMaker
// 
/**\class ComponentMaker ComponentMaker.h FWCore/Framework/interface/ComponentMaker.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Wed May 25 16:56:05 EDT 2005
// $Id: ComponentMaker.h,v 1.15 2007/07/30 19:14:21 chrjones Exp $
//

// system include files
#include <string>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

// forward declarations

namespace edm {
   class ParameterSet;
   namespace eventsetup {
      class EventSetupProvider;
      class DataProxyProvider;
     
template <class T>
      class ComponentMakerBase {
public:
         virtual ~ComponentMakerBase() {}
         typedef typename T::base_type base_type;
         virtual boost::shared_ptr<base_type> addTo(EventSetupProvider& iProvider,
                     ParameterSet const& iConfiguration,
                     std::string const& iProcessName,
                     ReleaseVersion const& iVersion,
                     PassID const& iPass) const = 0;
      };
      
template <class T, class TComponent>
   class ComponentMaker : public ComponentMakerBase<T>
{

   public:
   ComponentMaker() {}
      //virtual ~ComponentMaker();
   typedef typename T::base_type base_type;

      // ---------- const member functions ---------------------
   virtual boost::shared_ptr<base_type> addTo(EventSetupProvider& iProvider,
                       ParameterSet const& iConfiguration,
                       std::string const& iProcessName,
                       ReleaseVersion const& iVersion,
                       PassID const& iPass) const;
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      ComponentMaker(const ComponentMaker&); // stop default

      const ComponentMaker& operator=(const ComponentMaker&); // stop default

      void setDescription(DataProxyProvider* iProv, const ComponentDescription& iDesc) const {
        iProv->setDescription(iDesc);
      }
      void setDescriptionForFinder(EventSetupRecordIntervalFinder* iFinder, const ComponentDescription& iDesc) const {
        iFinder->setDescriptionForFinder(iDesc);
      }
      void setDescription(void*, const ComponentDescription&) const {
      }
      void setDescriptionForFinder(void*, const ComponentDescription&) const {
      }
      // ---------- member data --------------------------------

};

template< class T, class TComponent>
boost::shared_ptr<typename ComponentMaker<T,TComponent>::base_type>
ComponentMaker<T,TComponent>:: addTo(EventSetupProvider& iProvider,
                                        ParameterSet const& iConfiguration,
                                        std::string const& iProcessName,
                                        ReleaseVersion const& iVersion,
                                        PassID const& iPass) const
{
   boost::shared_ptr<TComponent> component(new TComponent(iConfiguration));
   
   ComponentDescription description;
   description.type_  = iConfiguration.template getParameter<std::string>("@module_type");
   description.label_ = iConfiguration.template getParameter<std::string>("@module_label");

   description.releaseVersion_ = iVersion;
   description.pid_           = iConfiguration.id();
   description.processName_   = iProcessName;
   description.passID_          = iPass;
      
   this->setDescription(component.get(),description);
   this->setDescriptionForFinder(component.get(),description);
   T::addTo(iProvider, component);
   return component;
}
   }
}
#endif
