// -*- C++ -*-
//
// Package:     Vertexs
// Class  :     FWVertexProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWVertexProxyBuilder.cc,v 1.2 2009/01/23 21:35:47 amraktad Exp $
//

// system include files
#include "TEvePointSet.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class FWVertexProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Vertex> {

public:
   FWVertexProxyBuilder() {}
   virtual ~FWVertexProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWVertexProxyBuilder(const FWVertexProxyBuilder&); // stop default
   const FWVertexProxyBuilder& operator=(const FWVertexProxyBuilder&); // stop default

   virtual void build(const reco::Vertex& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
};

//
// member functions
//
void
FWVertexProxyBuilder::build(const reco::Vertex& iData, unsigned int iIndex, TEveElement& oItemHolder) const
{
   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetMainColor( item()->defaultDisplayProperties().color() );
   pointSet->SetNextPoint( iData.x(), iData.y(), iData.z() );
   oItemHolder.AddElement( pointSet );
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWVertexProxyBuilder, reco::VertexCollection, "Vertices", FWViewType::k3DBit | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
