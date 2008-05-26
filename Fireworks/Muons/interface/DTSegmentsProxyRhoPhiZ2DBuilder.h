#ifndef Fireworks_Muons_DTSegmentsProxyRhoPhiZ2DBuilder_h
#define Fireworks_Muons_DTSegmentsProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Muons
// Class  :     DTSegmentsProxyRhoPhiZ2DBuilder
// 
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: DTSegmentsProxyRhoPhiZ2DBuilder.h,v 1.3 2008/05/13 05:23:43 dmytro Exp $
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

class DTSegmentsProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      DTSegmentsProxyRhoPhiZ2DBuilder();
      virtual ~DTSegmentsProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      static void build(const FWEventItem* iItem, 
			TEveElementList** product,
			bool rhoPhiProjection);
      // ---------- member functions ---------------------------
   
   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);
   
      virtual void buildRhoZ(const FWEventItem* iItem, 
                               TEveElementList** product);

   
      DTSegmentsProxyRhoPhiZ2DBuilder(const DTSegmentsProxyRhoPhiZ2DBuilder&); // stop default

      const DTSegmentsProxyRhoPhiZ2DBuilder& operator=(const DTSegmentsProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
