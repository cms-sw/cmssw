#ifndef Fireworks_Calo_ElectronsProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_ElectronsProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     ElectronsProxyRhoPhiZ2DBuilder
// 
/**\class ElectronsProxyRhoPhiZ2DBuilder ElectronsProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/ElectronsProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: ElectronsProxyRhoPhiZ2DBuilder.h,v 1.1 2008/02/11 19:09:17 jmuelmen Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"

// forward declarations

class TEveGeoShapeExtract;

class ElectronsProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      ElectronsProxyRhoPhiZ2DBuilder();
      virtual ~ElectronsProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);
   
      virtual void buildRhoZ(const FWEventItem* iItem, 
                               TEveElementList** product);

   private:
      ElectronsProxyRhoPhiZ2DBuilder(const ElectronsProxyRhoPhiZ2DBuilder&); // stop default

      const ElectronsProxyRhoPhiZ2DBuilder& operator=(const ElectronsProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
