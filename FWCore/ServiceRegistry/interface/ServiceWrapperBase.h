#ifndef ServiceRegistry_ServiceWrapperBase_h
#define ServiceRegistry_ServiceWrapperBase_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceWrapperBase
// 
/**\class ServiceWrapperBase ServiceWrapperBase.h FWCore/ServiceRegistry/interface/ServiceWrapperBase.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:01 EDT 2005
// $Id: ServiceWrapperBase.h,v 1.1 2005/09/07 21:58:16 chrjones Exp $
//

// system include files

// user include files

// forward declarations
namespace edm {
   namespace serviceregistry {
      
      class ServiceWrapperBase
      {

public:
         ServiceWrapperBase();
         virtual ~ServiceWrapperBase();

         // ---------- const member functions ---------------------
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         
private:
         ServiceWrapperBase(const ServiceWrapperBase&); // stop default
         
         const ServiceWrapperBase& operator=(const ServiceWrapperBase&); // stop default
         
         // ---------- member data --------------------------------
         
      };
   }
}

#endif
