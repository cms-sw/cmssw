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
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

// forward declarations

namespace edm {
   namespace eventsetup {
      class EventSetupProvider;
      class DataProxyProvider;
    
      class ComponentMakerBaseHelper
      {
      public:
        virtual ~ComponentMakerBaseHelper() {}
      protected:
        ComponentDescription createComponentDescription(ParameterSet const& iConfiguration,
                                                       std::string const& iProcessName,
                                                       ReleaseVersion const& iVersion,
                                                       PassID const& iPass) const;
      };
 
      template <class T>
      class ComponentMakerBase : private ComponentMakerBaseHelper {
      public:
         typedef typename T::base_type base_type;
         virtual boost::shared_ptr<base_type> addTo(EventSetupProvider& iProvider,
                     ParameterSet const& iConfiguration,
                     std::string const& iProcessName,
                     ReleaseVersion const& iVersion,
                     PassID const& iPass) const = 0;
      protected:
	using ComponentMakerBaseHelper::createComponentDescription;
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
      void setPostConstruction(DataProxyProvider* iProv, const edm::ParameterSet& iPSet) const {
        //The 'appendToDataLabel' parameter was added very late in the development cycle and since
        // the ParameterSet is not sent to the base class we must set the value after construction
        iProv->setAppendToDataLabel(iPSet);
      }
      void setDescription(void*, const ComponentDescription&) const {
      }
      void setDescriptionForFinder(void*, const ComponentDescription&) const {
      }
      void setPostConstruction(void*, const edm::ParameterSet&) const {
      }
      // ---------- member data --------------------------------

};

template< class T, class TComponent>
boost::shared_ptr<typename ComponentMaker<T,TComponent>::base_type>
ComponentMaker<T,TComponent>::addTo(EventSetupProvider& iProvider,
                                        ParameterSet const& iConfiguration,
                                        std::string const& iProcessName,
                                        ReleaseVersion const& iVersion,
                                        PassID const& iPass) const
{
   boost::shared_ptr<TComponent> component(new TComponent(iConfiguration));
   ComponentDescription description =
       this->createComponentDescription(iConfiguration,
                                        iProcessName,
                                        iVersion,
                                        iPass);
      
   this->setDescription(component.get(),description);
   this->setDescriptionForFinder(component.get(),description);
   this->setPostConstruction(component.get(),iConfiguration);
   T::addTo(iProvider, component);
   return component;
}
   }
}
#endif
