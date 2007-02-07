#ifndef ServiceRegistry_ServicePluginFactory_h
#define ServiceRegistry_ServicePluginFactory_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServicePluginFactory
// 
/**\class ServicePluginFactory ServicePluginFactory.h FWCore/ServiceRegistry/interface/ServicePluginFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:00 EDT 2005
// $Id: ServicePluginFactory.h,v 1.1 2005/09/07 21:58:16 chrjones Exp $
//

// system include files
#include "FWCore/PluginManager/interface/PluginFactory.h"

// user include files

// forward declarations
namespace edm {
   namespace serviceregistry {
      class ServiceMakerBase;
      
      class ServicePluginFactory :  public seal::PluginFactory< ServiceMakerBase* ()>
      {

public:
         ServicePluginFactory();
         //virtual ~ServicePluginFactory();

         // ---------- const member functions ---------------------
         
         // ---------- static member functions --------------------
         static ServicePluginFactory* get();
         
         // ---------- member functions ---------------------------
         
private:
         ServicePluginFactory(const ServicePluginFactory&); // stop default
         
         const ServicePluginFactory& operator=(const ServicePluginFactory&); // stop default
         
         // ---------- member data --------------------------------
         
};

   }
}

#endif
