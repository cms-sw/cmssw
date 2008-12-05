// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWMuonsGlimpseProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Dec  4 19:04:30 EST 2008
// $Id$
//

// system include files
#include "DataFormats/MuonReco/interface/Muon.h"

// user include files
#include "Fireworks/Core/interface/FWGlimpseSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

class FWMuonsGlimpseProxyBuilder : public FWGlimpseSimpleProxyBuilderTemplate<reco::Muon> {
   
public:
   FWMuonsGlimpseProxyBuilder();
   virtual ~FWMuonsGlimpseProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWMuonsGlimpseProxyBuilder(const FWMuonsGlimpseProxyBuilder&); // stop default
   
   const FWMuonsGlimpseProxyBuilder& operator=(const FWMuonsGlimpseProxyBuilder&); // stop default
   
   virtual void build(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const;

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
FWMuonsGlimpseProxyBuilder::FWMuonsGlimpseProxyBuilder()
{
}

// FWMuonsGlimpseProxyBuilder::FWMuonsGlimpseProxyBuilder(const FWMuonsGlimpseProxyBuilder& rhs)
// {
//    // do actual copying here;
// }

FWMuonsGlimpseProxyBuilder::~FWMuonsGlimpseProxyBuilder()
{
}

//
// assignment operators
//
// const FWMuonsGlimpseProxyBuilder& FWMuonsGlimpseProxyBuilder::operator=(const FWMuonsGlimpseProxyBuilder& rhs)
// {
//   //An exception safe implementation is
//   FWMuonsGlimpseProxyBuilder temp(rhs);
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
FWMuonsGlimpseProxyBuilder::build(const reco::Muon& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   FWEveScalableStraightLineSet* marker = new FWEveScalableStraightLineSet("","");
   marker->SetLineWidth(2);
   fw::addStraightLineSegment( marker, &iData, 1.0 );
   oItemHolder.AddElement(marker);
   //add to scaler at end so that it can scale the line after all ends have been added
   scaler()->addElement(marker);   
}

//
// static member functions
//
REGISTER_FWGLIMPSEDATAPROXYBUILDER(FWMuonsGlimpseProxyBuilder,std::vector<reco::Muon>,"Muons");
