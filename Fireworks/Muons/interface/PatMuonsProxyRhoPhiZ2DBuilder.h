#ifndef Fireworks_Muons_PatMuonProxyRhoPhiZ2DBuilder_h
#define Fireworks_Muons_PatMuonProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Muons
// Class  :     PatMuonsProxyRhoPhiZ2DBuilder
//
/**\class PatMuonsProxyRhoPhiZ2DBuilder PatMuonsProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/PatMuonsProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: PatMuonsProxyRhoPhiZ2DBuilder.h,v 1.1 2008/09/26 07:15:41 dmytro Exp $
//

#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"

class PatMuonsProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      PatMuonsProxyRhoPhiZ2DBuilder();
      virtual ~PatMuonsProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

      static void build(const FWEventItem* iItem,
			TEveElementList** product,
			bool showEndcap,
			bool onlyTracks = false);

      PatMuonsProxyRhoPhiZ2DBuilder(const PatMuonsProxyRhoPhiZ2DBuilder&); // stop default

      const PatMuonsProxyRhoPhiZ2DBuilder& operator=(const PatMuonsProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
