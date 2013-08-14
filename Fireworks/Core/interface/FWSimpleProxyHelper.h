#ifndef Fireworks_Core_FWSimpleProxyHelper_h
#define Fireworks_Core_FWSimpleProxyHelper_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWSimpleProxyHelper
//
/**\class FWSimpleProxyHelper FWSimpleProxyHelper.h Fireworks/Core/interface/FWSimpleProxyHelper.h

   Description: Implements some common functionality needed by all Simple ProxyBuilders

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 15:13:17 EST 2008
// $Id: FWSimpleProxyHelper.h,v 1.3 2010/08/18 10:30:10 amraktad Exp $
//

// system include files
#include <typeinfo>
#include <string>

// user include files

// forward declarations
class FWEventItem;

class FWSimpleProxyHelper {

public:
   FWSimpleProxyHelper(const std::type_info& );
   //virtual ~FWSimpleProxyHelper();

   // ---------- const member functions ---------------------
   const void* offsetObject(const void* iObj) const {
      return static_cast<const char*> (iObj)+m_objectOffset;
   }
   
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void itemChanged(const FWEventItem*);
private:
   //FWSimpleProxyHelper(const FWSimpleProxyHelper&); // stop default

   //const FWSimpleProxyHelper& operator=(const FWSimpleProxyHelper&); // stop default

   // ---------- member data --------------------------------
   const std::type_info* m_itemType;
   unsigned int m_objectOffset;

};


#endif
