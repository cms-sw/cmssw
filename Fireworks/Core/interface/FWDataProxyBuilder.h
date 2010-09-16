#ifndef Fireworks_Core_FWDataProxyBuilder_h
#define Fireworks_Core_FWDataProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDataProxyBuilder
//
/**\class FWDataProxyBuilder FWDataProxyBuilder.h Fireworks/Core/interface/FWDataProxyBuilder.h

   Description: Builds Proxies of Data used by a View

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Thu Dec  6 17:49:40 PST 2007
// $Id: FWDataProxyBuilder.h,v 1.5 2008/11/06 22:05:22 amraktad Exp $
//

// system include files
#include <string>

// user include files

// forward declarations
namespace fwlite {
   class Event;
}

class TEveElementList;
class TObject;
class FWEventItem;

class FWDataProxyBuilder
{

public:
   FWDataProxyBuilder();
   virtual ~FWDataProxyBuilder();

   // ---------- const member functions ---------------------
   //The 'main' C++ class type used by this proxy builder
   virtual const std::string typeName() const =0;
   //If this same C++ class is used for multiple purposes, designate to
   // which purpose this builder is used.  If only one purpose leave it an empty string
   virtual const std::string purpose() const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setItem(const FWEventItem* iItem);
   void build(TObject** product);

private:
   virtual void build(const FWEventItem* iItem, TObject** product) = 0 ;

   FWDataProxyBuilder(const FWDataProxyBuilder&);    // stop default

   const FWDataProxyBuilder& operator=(const FWDataProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
   const FWEventItem* m_item;

};


#endif
