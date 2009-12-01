// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWRecoMetLegoEveProxyBuilder

//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWRecoMetLegoEveProxyBuilder.cc,v 1.1 2009/05/13 05:02:39 dmytro Exp $
//

// system include files
#include "TEveElement.h"
#include "TEveStraightLineSet.h"
#include "TEveCompound.h"

// user include files
#include "Fireworks/Core/interface/FW3DLegoSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/MET.h"

class FWRecoMetLegoEveProxyBuilder : public FW3DLegoSimpleProxyBuilderTemplate<reco::MET>
{
public:
   FWRecoMetLegoEveProxyBuilder();
   virtual ~FWRecoMetLegoEveProxyBuilder();

   // ---------- const member functions ---------------------
   REGISTER_PROXYBUILDER_METHODS();

   // ---------- static member functions --------------------
private:
   void build(const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   FWRecoMetLegoEveProxyBuilder(const FWRecoMetLegoEveProxyBuilder&);    // stop default

   const FWRecoMetLegoEveProxyBuilder& operator=(const FWRecoMetLegoEveProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWRecoMetLegoEveProxyBuilder::FWRecoMetLegoEveProxyBuilder()
{
}

// FWRecoMetLegoEveProxyBuilder::FWRecoMetLegoEveProxyBuilder(const FWRecoMetLegoEveProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWRecoMetLegoEveProxyBuilder::~FWRecoMetLegoEveProxyBuilder()
{
}

//
// assignment operators
//
// const FWRecoMetLegoEveProxyBuilder& FWRecoMetLegoEveProxyBuilder::operator=(const FWRecoMetLegoEveProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWRecoMetLegoEveProxyBuilder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }


void
FWRecoMetLegoEveProxyBuilder::build(const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{

	TEveStraightLineSet* mainLine = new TEveStraightLineSet( "MET phi" );
	// mainLine->SetLineWidth(2);
	mainLine->AddLine(-5.191, iData.phi(), 0.1, 5.191, iData.phi(), 0.1 );
	oItemHolder.AddElement( mainLine );

	double phi = iData.phi();
	phi = phi > 0 ? phi - M_PI : phi + M_PI;
	TEveStraightLineSet* secondLine = new TEveStraightLineSet( "MET opposite phi" );
	// secondLine->SetLineWidth(2);
	secondLine->SetLineStyle(7);
	secondLine->AddLine(-5.191, phi, 0.1, 5.191, phi, 0.1 );
	oItemHolder.AddElement( secondLine );

}

REGISTER_FW3DLEGODATAPROXYBUILDER(FWRecoMetLegoEveProxyBuilder,reco::MET,"recoMET");
