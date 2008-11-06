#ifndef Fireworks_Calo_PatJetProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_PatJetProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     PatJetProxyRhoPhiZ2DBuilder
//
/**\class PatJetProxyRhoPhiZ2DBuilder PatJetProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/PatJetProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: PatJetProxyRhoPhiZ2DBuilder.h,v 1.1 2008/09/26 07:40:12 dmytro Exp $
//

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"

// forward declarations

class TEveGeoShapeExtract;
class PatJetProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      PatJetProxyRhoPhiZ2DBuilder();
      virtual ~PatJetProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

      PatJetProxyRhoPhiZ2DBuilder(const PatJetProxyRhoPhiZ2DBuilder&); // stop default

      const PatJetProxyRhoPhiZ2DBuilder& operator=(const PatJetProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
