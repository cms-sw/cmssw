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
// $Id: ComponentMaker.h,v 1.5 2005/08/24 21:43:24 chrjones Exp $
//

// system include files
#include <string>
#include "boost/shared_ptr.hpp"

// user include files

// forward declarations

namespace edm {
   class ParameterSet;
   namespace eventsetup {
      class EventSetupProvider;
      
template <class T>
      class ComponentMakerBase {
public:
         virtual void addTo(EventSetupProvider& iProvider,
                     ParameterSet const& iConfiguration,
                     std::string const& iProcessName,
                     unsigned long iVersion,
                     unsigned long iPass) const = 0;
      };
      
template <class T, class TComponent>
   class ComponentMaker : public ComponentMakerBase<T>
{

   public:
   ComponentMaker() {}
      //virtual ~ComponentMaker();

      // ---------- const member functions ---------------------
   virtual void addTo(EventSetupProvider& iProvider,
                       ParameterSet const& iConfiguration,
                       std::string const& iProcessName,
                       unsigned long iVersion,
                       unsigned long iPass) const;
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      ComponentMaker(const ComponentMaker&); // stop default

      const ComponentMaker& operator=(const ComponentMaker&); // stop default

      // ---------- member data --------------------------------

};

template< class T, class TComponent>
void
ComponentMaker<T,TComponent>:: addTo(EventSetupProvider& iProvider,
                                        ParameterSet const& iConfiguration,
                                        std::string const& /*iProcessName*/,
                                        unsigned long /*iVersion*/,
                                        unsigned long /*iPass*/) const 
{
   boost::shared_ptr<TComponent> component(new TComponent(iConfiguration));
   
   T::addTo(iProvider, component);
}
   }
}
#endif /* Framework_ComponentMaker_h */
