/*
 *  FWSimVertexProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/9/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "TEvePointSet.h"

class FWSimVertexProxyBuilder : public FWSimpleProxyBuilderTemplate<SimVertex>
{
public:
   FWSimVertexProxyBuilder( void ) {} 
   virtual ~FWSimVertexProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWSimVertexProxyBuilder( const FWSimVertexProxyBuilder& );
   // Disable default assignment operator
   const FWSimVertexProxyBuilder& operator=( const FWSimVertexProxyBuilder& );

   void build( const SimVertex& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWSimVertexProxyBuilder::build( const SimVertex& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* )
{
   TEvePointSet* pointSet = new TEvePointSet;
   setupAddElement( pointSet, &oItemHolder );
   pointSet->SetNextPoint( iData.position().x() * 0.01, iData.position().y() * 0.01, iData.position().z() * 0.01 );
}

REGISTER_FWPROXYBUILDER( FWSimVertexProxyBuilder, SimVertex, "SimVertices", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
