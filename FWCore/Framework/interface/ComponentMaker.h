#ifndef EVENTSETUP_COMPONENTMAKER_H
#define EVENTSETUP_COMPONENTMAKER_H
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
// $Id: ComponentMaker.h,v 1.3 2005/06/23 19:59:30 wmtan Exp $
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
                                        std::string const& iProcessName,
                                        unsigned long iVersion,
                                        unsigned long iPass) const 
{
   boost::shared_ptr<TComponent> component(new TComponent(iConfiguration));
   
   T::addTo(iProvider, component);
}
   }
}
#endif /* EVENTSETUP_COMPONENTMAKER_H */
