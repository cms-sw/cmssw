#ifndef Fireworks_Core_FWRPZ2DDataProxyBuilder_h
#define Fireworks_Core_FWRPZ2DDataProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZ2DDataProxyBuilder
// 
/**\class FWRPZ2DDataProxyBuilder FWRPZ2DDataProxyBuilder.h Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sat Jan  5 15:02:03 EST 2008
// $Id: FWRPZ2DDataProxyBuilder.h,v 1.1 2008/01/07 05:48:45 chrjones Exp $
//

// system include files

// user include files

// forward declarations
class FWEventItem;
class TEveElementList;

class FWRPZ2DDataProxyBuilder
{

   public:
      FWRPZ2DDataProxyBuilder();
      virtual ~FWRPZ2DDataProxyBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setItem(const FWEventItem* iItem);
      void buildRhoPhi(TEveElementList** product);
      void buildRhoZ(TEveElementList** product);

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem, 
                               TEveElementList** product) = 0 ;
      virtual void buildRhoZ(const FWEventItem* iItem, 
                               TEveElementList** product) = 0 ;

      FWRPZ2DDataProxyBuilder(const FWRPZ2DDataProxyBuilder&); // stop default

      const FWRPZ2DDataProxyBuilder& operator=(const FWRPZ2DDataProxyBuilder&); // stop default

      // ---------- member data --------------------------------
      const FWEventItem* m_item;

};


#endif
