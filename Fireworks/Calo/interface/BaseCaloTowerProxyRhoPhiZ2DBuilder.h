#ifndef Fireworks_Calo_BaseCaloTowerProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_BaseCaloTowerProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     BaseCaloTowerProxyRhoPhiZ2DBuilder
// 
/**\class BaseCaloTowerProxyRhoPhiZ2DBuilder BaseCaloTowerProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/BaseCaloTowerProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Mar 12 21:32:49 CDT 2008
// $Id$
//

// system include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"

// user include files

// forward declarations

class BaseCaloTowerProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      BaseCaloTowerProxyRhoPhiZ2DBuilder();
      virtual ~BaseCaloTowerProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
protected:
      //need to special handle selection or other changes
      virtual void modelChangesRhoPhi(const FWModelIds&, TEveElement*);
      virtual void modelChangesRhoZ(const FWModelIds&, TEveElement*);
   
   
   private:
      BaseCaloTowerProxyRhoPhiZ2DBuilder(const BaseCaloTowerProxyRhoPhiZ2DBuilder&); // stop default

      const BaseCaloTowerProxyRhoPhiZ2DBuilder& operator=(const BaseCaloTowerProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------

};


#endif
