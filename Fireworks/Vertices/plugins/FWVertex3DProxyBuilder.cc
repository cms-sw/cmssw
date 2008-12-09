// -*- C++ -*-
//
// Package:     Vertexs
// Class  :     FWVertex3DProxyBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWVertex3DProxyBuilder.cc,v 1.1 2008/12/04 15:26:01 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TEvePointSet.h"

class FWVertex3DProxyBuilder: public FW3DSimpleProxyBuilderTemplate<reco::Vertex> {
      
public:
   FWVertex3DProxyBuilder();
   //virtual ~FWVertex3DProxyBuilder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   REGISTER_PROXYBUILDER_METHODS();
 
private:
   FWVertex3DProxyBuilder(const FWVertex3DProxyBuilder&); // stop default
   
   const FWVertex3DProxyBuilder& operator=(const FWVertex3DProxyBuilder&); // stop default
   
   virtual void build(const reco::Vertex& iData, unsigned int iIndex,TEveElement& oItemHolder) const;
   
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
FWVertex3DProxyBuilder::FWVertex3DProxyBuilder()
{
}


//
// member functions
//
void 
FWVertex3DProxyBuilder::build(const reco::Vertex& iData, unsigned int iIndex,TEveElement& oItemHolder) const
{
   TEvePointSet* pointSet = new TEvePointSet();
   pointSet->SetMainColor(item()->defaultDisplayProperties().color());
   pointSet->SetNextPoint( iData.x(), iData.y(), iData.z() );
   oItemHolder.AddElement( pointSet );
}


//
// const member functions
//

//
// static member functions
//

REGISTER_FW3DDATAPROXYBUILDER(FWVertex3DProxyBuilder,reco::VertexCollection,"Vertices");
