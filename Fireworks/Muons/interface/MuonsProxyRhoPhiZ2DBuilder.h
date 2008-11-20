#ifndef Fireworks_Calo_CaloJetProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_CaloJetProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Muons
// Class  :     MuonsProxyRhoPhiZ2DBuilder
//
/**\class MuonsProxyRhoPhiZ2DBuilder MuonsProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/MuonsProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: MuonsProxyRhoPhiZ2DBuilder.h,v 1.9 2008/11/06 22:05:29 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "TEveVSDStructs.h"

#include "Fireworks/Muons/interface/FWMuonBuilder.h"

// forward declarations
namespace reco
{
   class Muon;
   class TrackExtra;
}
namespace fw
{
   class NamedCounter;
}

class TEveTrack;
class TEveElementList;
class TEveGeoShapeExtract;
class DetIdToMatrix;

class MuonsProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      MuonsProxyRhoPhiZ2DBuilder();
      virtual ~MuonsProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

      static bool buggyMuon( const reco::Muon* muon,
			     const DetIdToMatrix* geom );

      void build(const FWEventItem* iItem,
                 TEveElementList** product,
                 bool showEndcap,
                 bool onlyTracks = false);

      MuonsProxyRhoPhiZ2DBuilder(const MuonsProxyRhoPhiZ2DBuilder&); // stop default

      const MuonsProxyRhoPhiZ2DBuilder& operator=(const MuonsProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
      FWMuonBuilder m_builder;
};


#endif
