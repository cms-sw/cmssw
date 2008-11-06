#ifndef Fireworks_Muons_RPCActiveChamberProxyRhoPhiZ2DBuilder_h
#define Fireworks_Muons_RPCActiveChamberProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Muons
// Class  :     RPCActiveChamberProxyRhoPhiZ2DBuilder
//
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: RPCActiveChamberProxyRhoPhiZ2DBuilder.h,v 1.1 2008/08/24 13:19:03 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

// forward declarations
namespace reco
{
   class Muon;
   class TrackExtra;
}

class TEveTrack;
class TEveElementList;
class TEveGeoShapeExtract;
class DetIdToMatrix;

class RPCActiveChamberProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      RPCActiveChamberProxyRhoPhiZ2DBuilder();
      virtual ~RPCActiveChamberProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      static void build(const FWEventItem* iItem,
			TEveElementList** product,
			bool rhoPhiProjection);
      // ---------- member functions ---------------------------
      REGISTER_PROXYBUILDER_METHODS();

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void modelChanges(const FWModelIds& iIds,
				TEveElement* iElements);
      virtual void applyChangesToAllModels(TEveElement* iElements);

      RPCActiveChamberProxyRhoPhiZ2DBuilder(const RPCActiveChamberProxyRhoPhiZ2DBuilder&); // stop default

      const RPCActiveChamberProxyRhoPhiZ2DBuilder& operator=(const RPCActiveChamberProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
