// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWRecoMet3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWRecoMet3DProxyBuilder.cc,v 1.2 2009/05/18 01:42:47 dmytro Exp $
//

// system include files
#include "TEveManager.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveElement.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FW3DView.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"

#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/MET.h"

class FWRecoMet3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::MET>
{

public:
   FWRecoMet3DProxyBuilder();
   virtual ~FWRecoMet3DProxyBuilder();

   // ---------- const member functions ---------------------
   REGISTER_PROXYBUILDER_METHODS();

   // ---------- static member functions --------------------
private:
   virtual void build(const reco::MET& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   FWRecoMet3DProxyBuilder(const FWRecoMet3DProxyBuilder&);    // stop default

   const FWRecoMet3DProxyBuilder& operator=(const FWRecoMet3DProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWRecoMet3DProxyBuilder::FWRecoMet3DProxyBuilder()
{
}

FWRecoMet3DProxyBuilder::~FWRecoMet3DProxyBuilder()
{
}

void
FWRecoMet3DProxyBuilder::build(const reco::MET& iData, unsigned int iIndex,TEveElement& oItemHolder) const

{
   double r_ecal = 126;
   
	double phi = iData.phi();
      // double min_phi = phi-M_PI/36/2;
      // double max_phi = phi+M_PI/36/2;

     
      // double size = mets->at(i).et();
	double size = iData.et()*2;

      // TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
      // TEveGeoShape *element = fw::getShape( "spread", sc_box, iItem->defaultDisplayProperties().color() );
      // element->SetPickable(kTRUE);
      // container->AddElement(element);

	TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
	marker->SetLineWidth(2);
      // marker->SetLineStyle(kDotted);
	marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
    //   const double dx = 0.9*size*0.05;
//       const double dy = 0.9*size*cos(0.05);
	const double dx = 0.9*size*0.1;
	const double dy = 0.9*size*cos(0.1);
	marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0,
                       (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
	marker->AddLine( dx*sin(phi) + (dy+r_ecal)*cos(phi), -dx*cos(phi) + (dy+r_ecal)*sin(phi), 0,
                       (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
	marker->AddLine( -dx*sin(phi) + (dy+r_ecal)*cos(phi), dx*cos(phi) + (dy+r_ecal)*sin(phi), 0,
                       (r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);

	oItemHolder.AddElement(marker);

}

REGISTER_FW3DDATAPROXYBUILDER(FWRecoMet3DProxyBuilder,reco::MET,"recoMET");

