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
// $Id: FWRPZ2DDataProxyBuilder.h,v 1.2 2008/01/28 14:02:25 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWModelChangeSignal.h"

// forward declarations
class FWEventItem;
class TEveElementList;
class TEveElement;

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

      void modelChangesRhoPhi(const FWModelIds&);
      void modelChangesRhoZ(const FWModelIds&);
   
      void setRhoPhiProj(TEveElement*);
      void setRhoZProj(TEveElement*);
   
   protected:
      virtual void buildRhoPhi(const FWEventItem* iItem, 
                               TEveElementList** product) = 0 ;
      virtual void buildRhoZ(const FWEventItem* iItem, 
                               TEveElementList** product) = 0 ;

      //Override this if you need to special handle selection or other changes
      virtual void modelChangesRhoPhi(const FWModelIds&, TEveElement*);
      virtual void modelChangesRhoZ(const FWModelIds&, TEveElement*);

      FWRPZ2DDataProxyBuilder(const FWRPZ2DDataProxyBuilder&); // stop default

      const FWRPZ2DDataProxyBuilder& operator=(const FWRPZ2DDataProxyBuilder&); // stop default

      // ---------- member data --------------------------------
      const FWEventItem* m_item;

      TEveElementList* m_rhoPhiElements;
      TEveElementList* m_rhoPhiZElements;

      TEveElement* m_rhoPhiProj;
      TEveElement* m_rhoZProj;

};


#endif
