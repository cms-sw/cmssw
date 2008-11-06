#ifndef Fireworks_Muons_CSCSegmentsProxyRhoPhiZ2DBuilder_h
#define Fireworks_Muons_CSCSegmentsProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Muons
// Class  :     CSCSegmentsProxyRhoPhiZ2DBuilder
//
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: CSCSegmentsProxyRhoPhiZ2DBuilder.h,v 1.2 2008/07/15 18:20:46 dmytro Exp $
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

class CSCSegmentsProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      CSCSegmentsProxyRhoPhiZ2DBuilder();
      virtual ~CSCSegmentsProxyRhoPhiZ2DBuilder();

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

      CSCSegmentsProxyRhoPhiZ2DBuilder(const CSCSegmentsProxyRhoPhiZ2DBuilder&); // stop default

      const CSCSegmentsProxyRhoPhiZ2DBuilder& operator=(const CSCSegmentsProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
