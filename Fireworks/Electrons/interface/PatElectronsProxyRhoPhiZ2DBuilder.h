#ifndef Fireworks_Calo_PatElectronsProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_PatElectronsProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     PatElectronsProxyRhoPhiZ2DBuilder
//
/**\class PatElectronsProxyRhoPhiZ2DBuilder PatElectronsProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/PatElectronsProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: PatElectronsProxyRhoPhiZ2DBuilder.h,v 1.1 2008/09/26 07:42:08 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"

// forward declarations

class TEveGeoShapeExtract;

class PatElectronsProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      PatElectronsProxyRhoPhiZ2DBuilder();
      virtual ~PatElectronsProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

   private:
      PatElectronsProxyRhoPhiZ2DBuilder(const PatElectronsProxyRhoPhiZ2DBuilder&); // stop default

      const PatElectronsProxyRhoPhiZ2DBuilder& operator=(const PatElectronsProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
