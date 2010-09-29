// -*- C++ -*-
//
// Package:     CaloJets
// Class  :     FWCaloJet3DProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWCaloJet3DProxyBuilder.cc,v 1.3 2009/01/23 21:35:39 amraktad Exp $
//

// system include files
#include "DataFormats/JetReco/interface/Jet.h"

// user include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "Fireworks/Calo/interface/FW3DEveJet.h"


class FWCaloJet3DProxyBuilder : public FW3DSimpleProxyBuilderTemplate<reco::Jet> {

public:
   FWCaloJet3DProxyBuilder();
   //virtual ~FWCaloJet3DProxyBuilder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloJet3DProxyBuilder(const FWCaloJet3DProxyBuilder&); // stop default

   const FWCaloJet3DProxyBuilder& operator=(const FWCaloJet3DProxyBuilder&); // stop default

   virtual void build(const reco::Jet& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

   // ---------- member data --------------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
namespace {
   double getTheta( double eta ) {
      return 2*atan(exp(-eta));
   }
}

//
// constructors and destructor
//
FWCaloJet3DProxyBuilder::FWCaloJet3DProxyBuilder()
{
}

//
// member functions
//
void
FWCaloJet3DProxyBuilder::build(const reco::Jet& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   FW3DEveJet* cone = new FW3DEveJet(iData,"jet","jet");
   cone->SetPickable(kTRUE);
   cone->SetMainColor( oItemHolder.GetMainColor() );
   cone->SetMainTransparency(75);

   oItemHolder.AddElement( cone );
}


//
// const member functions
//

//
// static member functions
//

REGISTER_FW3DDATAPROXYBUILDER(FWCaloJet3DProxyBuilder,reco::Jet,"Jets");
