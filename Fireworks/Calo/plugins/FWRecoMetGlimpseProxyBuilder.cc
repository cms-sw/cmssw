// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWRecoMetGlimpseProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 23:57:00 EST 2008
// $Id: FWRecoMetGlimpseProxyBuilder.cc,v 1.1 2009/05/13 05:02:39 dmytro Exp $
//

// system include files
#include "TEveElement.h"

// user include files
#include "Fireworks/Core/interface/FWGlimpseSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"

#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/MET.h"

class FWRecoMetGlimpseProxyBuilder : public FWGlimpseSimpleProxyBuilderTemplate<reco::MET>
{

public:
   FWRecoMetGlimpseProxyBuilder();
   virtual ~FWRecoMetGlimpseProxyBuilder();

   // ---------- const member functions ---------------------
   REGISTER_PROXYBUILDER_METHODS();

   // ---------- static member functions --------------------
private:
   virtual void build(const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder) const;

   FWRecoMetGlimpseProxyBuilder(const FWRecoMetGlimpseProxyBuilder&);    // stop default

   const FWRecoMetGlimpseProxyBuilder& operator=(const FWRecoMetGlimpseProxyBuilder&);    // stop default

   // ---------- member data --------------------------------
};

//
// constructors and destructor
//
FWRecoMetGlimpseProxyBuilder::FWRecoMetGlimpseProxyBuilder()
{
}

FWRecoMetGlimpseProxyBuilder::~FWRecoMetGlimpseProxyBuilder()
{
}

void
FWRecoMetGlimpseProxyBuilder::build(const reco::MET& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
	double phi = iData.phi();
	double size = iData.et();
	TEveScalableStraightLineSet* marker = new TEveScalableStraightLineSet("energy");
	marker->SetLineWidth(2);
	// marker->SetLineStyle(kDotted);
	marker->AddLine( 0, 0, 0, size*cos(phi), size*sin(phi), 0);
	marker->AddLine( size*0.9*cos(phi+0.03), size*0.9*sin(phi+0.03), 0, size*cos(phi), size*sin(phi), 0);
	marker->AddLine( size*0.9*cos(phi-0.03), size*0.9*sin(phi-0.03), 0, size*cos(phi), size*sin(phi), 0);
	oItemHolder.AddElement(marker);

}

REGISTER_FWGLIMPSEDATAPROXYBUILDER(FWRecoMetGlimpseProxyBuilder,reco::MET,"recoMET");

