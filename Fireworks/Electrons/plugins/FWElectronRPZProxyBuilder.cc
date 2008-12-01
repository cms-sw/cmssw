// -*- C++ -*-
//
// Package:     Electrons
// Class  :     FWElectronRPZProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 26 14:52:01 EST 2008
// $Id: FWElectronRPZProxyBuilder.cc,v 1.3 2008/11/29 03:01:47 dmytro Exp $
//

// system include files
#include "TEveGeoNode.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "TEveManager.h"
#include "TH1F.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEveTrack.h"
#include "TEveCompound.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"


// user include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

class FWElectronRPZProxyBuilder : public FWRPZ2DSimpleProxyBuilderTemplate<reco::GsfElectron> {
      
public:
   FWElectronRPZProxyBuilder();
   virtual ~FWElectronRPZProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();
   
private:
   FWElectronRPZProxyBuilder(const FWElectronRPZProxyBuilder&); // stop default
   
   const FWElectronRPZProxyBuilder& operator=(const FWElectronRPZProxyBuilder&); // stop default
   
   void buildRhoPhi(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   void buildRhoZ(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
      // ---------- member data --------------------------------
   
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWElectronRPZProxyBuilder::FWElectronRPZProxyBuilder()
{
}

// FWElectronRPZProxyBuilder::FWElectronRPZProxyBuilder(const FWElectronRPZProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWElectronRPZProxyBuilder::~FWElectronRPZProxyBuilder()
{
}

//
// assignment operators
//
// const FWElectronRPZProxyBuilder& FWElectronRPZProxyBuilder::operator=(const FWElectronRPZProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWElectronRPZProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//
void 
FWElectronRPZProxyBuilder::buildRhoPhi(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   if ( iData.superCluster().isAvailable() ) {
      TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());

      std::vector<DetId> detids = iData.superCluster()->getHitsByDetId();
      std::vector<double> phis;
      for (std::vector<DetId>::const_iterator id = detids.begin(); id != detids.end(); ++id) {
	 const TGeoHMatrix* matrix = item()->getGeom()->getMatrix( id->rawId() );
	 if ( matrix ) phis.push_back( atan2(matrix->GetTranslation()[1], matrix->GetTranslation()[0]) );
      }
      std::pair<double,double> phiRange = fw::getPhiRange( phis, iData.phi() );
      const double r = 122;
      TGeoBBox *sc_box = new TGeoTubeSeg(r - 1, r + 1, 1,
					 phiRange.first * 180 / M_PI - 0.5,
					 phiRange.second * 180 / M_PI + 0.5 ); // 0.5 is roughly half size of a crystal
      TEveGeoShape *sc = fw::getShape( "supercluster", sc_box, item()->defaultDisplayProperties().color() );
      sc->SetPickable(kTRUE);
      oItemHolder.AddElement(sc);
   }
   
   TEveTrack* track(0);
   if ( iData.gsfTrack().isAvailable() )
      track = fw::getEveTrack( *(iData.gsfTrack()) );
   else
      track = fw::getEveTrack( iData );
   oItemHolder.AddElement(track);   
}

void 
FWElectronRPZProxyBuilder::buildRhoZ(const reco::GsfElectron& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());

   if ( iData.superCluster().isAvailable() ) {
      double theta_max = 0;
      double theta_min = 10;
      std::vector<DetId> detids = iData.superCluster()->getHitsByDetId();
      for (std::vector<DetId>::const_iterator id = detids.begin(); id != detids.end(); ++id) {
	 const TGeoHMatrix* matrix = item()->getGeom()->getMatrix( id->rawId() );
	 if ( matrix ) {
	    double r = sqrt( matrix->GetTranslation()[0]*matrix->GetTranslation()[0] +
                            matrix->GetTranslation()[1]*matrix->GetTranslation()[1] );
	    double theta = atan2(r,matrix->GetTranslation()[2]);
	    if ( theta > theta_max ) theta_max = theta;
	    if ( theta < theta_min ) theta_min = theta;
	 }
      }
      // expand theta range by the size of a crystal to avoid segments of zero length
      double z_ecal = 302; // ECAL endcap inner surface
      double r_ecal = 122;
      fw::addRhoZEnergyProjection( &oItemHolder, r_ecal, z_ecal, theta_min-0.003, theta_max+0.003,
                                  iData.phi(), item()->defaultDisplayProperties().color() );
   }
   
   TEveTrack* track(0);
   if ( iData.gsfTrack().isAvailable() )
      track = fw::getEveTrack( *(iData.gsfTrack()) );
   else
      track = fw::getEveTrack( iData );
   oItemHolder.AddElement(track);
   
}

//
// static member functions
//
REGISTER_FWRPZDATAPROXYBUILDERBASE(FWElectronRPZProxyBuilder,reco::GsfElectron,"Electrons");
