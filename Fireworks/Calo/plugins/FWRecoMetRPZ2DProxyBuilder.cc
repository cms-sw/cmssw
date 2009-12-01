// -*- C++ -*-
//FWRecoMet
// Package:     Calo
// Class  :     FWRecoMetRPZ2DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWRecoMetRPZ2DProxyBuilder.cc,v 1.2 2009/01/23 21:35:40 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveScalableStraightLineSet.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FWRPZ2DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/METReco/interface/MET.h"

class FWRecoMetRPZ2DProxyBuilder : public FWRPZ2DSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWRecoMetRPZ2DProxyBuilder();
   virtual ~FWRecoMetRPZ2DProxyBuilder();

   // ---------- const member functions ---------------------
   REGISTER_PROXYBUILDER_METHODS();

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
	
   double getTheta( double eta ) {
      return 2*atan(exp(-eta));
   }

   void buildRhoPhi(const reco::MET& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   void buildRhoZ(const reco::MET& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   FWRecoMetRPZ2DProxyBuilder(const FWRecoMetRPZ2DProxyBuilder&);    // stop default

   const FWRecoMetRPZ2DProxyBuilder& operator=(const FWRecoMetRPZ2DProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWRecoMetRPZ2DProxyBuilder::FWRecoMetRPZ2DProxyBuilder()
{
}

// FWRecoMetRPZ2DProxyBuilder::FWRecoMetRPZ2DProxyBuilder(const FWRecoMetRPZ2DProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWRecoMetRPZ2DProxyBuilder::~FWRecoMetRPZ2DProxyBuilder()
{
}

//
// member functions
//
void
FWRecoMetRPZ2DProxyBuilder::buildRhoPhi(const reco::MET& iData, 
									unsigned int iIndex,
									TEveElement& oItemHolder) const
{
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   double r_ecal = 126;

	double	phi = iData.phi();
	double min_phi = phi-M_PI/36/2;
	double max_phi = phi+M_PI/36/2;

	double size = iData.et();
	TGeoBBox *sc_box = new TGeoTubeSeg(r_ecal - 1, r_ecal + 1, 1, min_phi * 180 / M_PI, max_phi * 180 / M_PI);
	TEveGeoShape *element = fw::getShape( "spread", sc_box, 0 );
	element->SetPickable(kTRUE);
	oItemHolder.AddElement(element);

	TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
	marker->SetLineWidth(2);

	marker->SetScaleCenter( r_ecal*cos(phi), r_ecal*sin(phi), 0 );
	const double dx = 0.9*size*0.05;
	const double dy = 0.9*size*cos(0.05);
	marker->AddLine( r_ecal*cos(phi), r_ecal*sin(phi), 0,
							(r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
	marker->AddLine( dx*sin(phi) + (dy+r_ecal)*cos(phi), -dx*cos(phi) + (dy+r_ecal)*sin(phi), 0,
							(r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);
	marker->AddLine( -dx*sin(phi) + (dy+r_ecal)*cos(phi), dx*cos(phi) + (dy+r_ecal)*sin(phi), 0,
							(r_ecal+size)*cos(phi), (r_ecal+size)*sin(phi), 0);

	oItemHolder.AddElement(marker);
}

void
FWRecoMetRPZ2DProxyBuilder::buildRhoZ(const reco::MET& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   double r = 126;

	double phi = iData.phi();
	double size = iData.et();

	TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
	marker->SetLineWidth(2);
	marker->SetScaleCenter(0., (phi>0 ? r : -r), 0);
	const double dx = 0.9*size*0.05;
	const double dy = 0.9*size*cos(0.05);
	marker->AddLine(0., (phi>0 ? r : -r), 0,
						 0., (phi>0 ? (r+size) : -(r+size)), 0 );
	marker->AddLine(0., (phi>0 ? r+dy : -(r+dy) ), dx,
						 0., (phi>0 ? (r+size) : -(r+size)), 0 );
	marker->AddLine(0., (phi>0 ? r+dy : -(r+dy) ), -dx,
						 0., (phi>0 ? (r+size) : -(r+size)), 0 );
	oItemHolder.AddElement( marker );
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWRecoMetRPZ2DProxyBuilder,reco::MET,"recoMET");
