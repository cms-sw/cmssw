#ifndef Fireworks_Core_FWRPZDataProxyBuilder_h
#define Fireworks_Core_FWRPZDataProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZDataProxyBuilder
// 
/**\class FWRPZDataProxyBuilder FWRPZDataProxyBuilder.h Fireworks/Core/interface/FWRPZDataProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sat Jan  5 15:02:03 EST 2008
// $Id$
//

// system include files

// user include files

// forward declarations
class FWEventItem;
class TEveElementList;

class FWRPZDataProxyBuilder
{

   public:
      FWRPZDataProxyBuilder();
      virtual ~FWRPZDataProxyBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setItem(const FWEventItem* iItem);
      void build(TEveElementList** product);

   private:
      virtual void build(const FWEventItem* iItem, 
			 TEveElementList** product) = 0 ;

      FWRPZDataProxyBuilder(const FWRPZDataProxyBuilder&); // stop default

      const FWRPZDataProxyBuilder& operator=(const FWRPZDataProxyBuilder&); // stop default

      // ---------- member data --------------------------------
      const FWEventItem* m_item;

};


#endif
