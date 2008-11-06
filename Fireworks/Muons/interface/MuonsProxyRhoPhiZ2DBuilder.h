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
// $Id: MuonsProxyRhoPhiZ2DBuilder.h,v 1.8 2008/09/26 07:15:41 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "TEveVSDStructs.h"
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

      static void addMatchInformation( const reco::Muon* muon,
				       const FWEventItem* iItem,
				       TEveTrack* track,
				       TEveElementList* parentList,
				       bool showEndcap,
				       bool onlyTracks = false);

      static TEveVector muonLocation( const reco::Muon* muon,
				      const FWEventItem* iItem );
      static TEveVector firstMatch( const reco::Muon* muon,
				    const FWEventItem* iItem );

      static void buildMuon(const FWEventItem* iItem,
			    const reco::Muon* muon,
			    TEveElementList* tList,
			    const fw::NamedCounter& counter,
			    bool showEndcap,
			    bool onlyTracks = false);

      // ---------- member functions ---------------------------

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

      static bool buggyMuon( const reco::Muon* muon,
			     const DetIdToMatrix* geom );

      static void build(const FWEventItem* iItem,
			TEveElementList** product,
			bool showEndcap,
			bool onlyTracks = false);

      MuonsProxyRhoPhiZ2DBuilder(const MuonsProxyRhoPhiZ2DBuilder&); // stop default

      const MuonsProxyRhoPhiZ2DBuilder& operator=(const MuonsProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
