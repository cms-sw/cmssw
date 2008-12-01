#ifndef Fireworks_Calo_Muons3DProxyBuilder_h
#define Fireworks_Calo_Muons3DProxyBuilder_h
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
// $Id: MuonsProxyRhoPhiZ2DBuilder.h,v 1.10 2008/11/20 01:15:28 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DDataProxyBuilder.h"
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

class Muons3DProxyBuilder : public FW3DDataProxyBuilder
{

   public:
      Muons3DProxyBuilder();
      virtual ~Muons3DProxyBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:

      void build(const FWEventItem* iItem,
                 TEveElementList** product);

      Muons3DProxyBuilder(const Muons3DProxyBuilder&); // stop default

      const Muons3DProxyBuilder& operator=(const Muons3DProxyBuilder&); // stop default

      // ---------- member data --------------------------------
      FWMuonBuilder m_builder;
};


#endif
