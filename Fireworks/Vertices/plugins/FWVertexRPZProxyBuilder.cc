// -*- C++ -*-
//
// Package:     Vertexs
// Class  :     FWVertexRPZProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
// $Id: FWVertexRPZProxyBuilder.cc,v 1.1 2009/01/16 17:09:38 amraktad Exp $
//

#include <cmath>

#include "RVersion.h"
#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEvePointSet.h"

// user include files
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

static const double scaleError  = 1.;

class FWVertexRPZProxyBuilder : public FWRPZDataProxyBuilder {
public:
   FWVertexRPZProxyBuilder();
   virtual ~FWVertexRPZProxyBuilder();

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build (const FWEventItem* item, TEveElementList** product);

   // prevent default copy constructor and assignment operator
   FWVertexRPZProxyBuilder (const FWVertexRPZProxyBuilder &);
   const FWVertexRPZProxyBuilder & operator=(const FWVertexRPZProxyBuilder &);
};

FWVertexRPZProxyBuilder::FWVertexRPZProxyBuilder() {
}

FWVertexRPZProxyBuilder::~FWVertexRPZProxyBuilder() {
}

void FWVertexRPZProxyBuilder::build(const FWEventItem* item, TEveElementList** product)
{
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TEveElementList * list = *product;

   if (list == NULL) {
      list = new TEveElementList(item->name().c_str(), "Primary Vertices", true);
      *product = list;
      list->SetMainColor(item->defaultDisplayProperties().color());
      gEve->AddElement(list);
   } else {
      list->DestroyElements();
   }

   const reco::VertexCollection * vertices;
   item->get(vertices);
   if (vertices == 0) {
      // std::cout <<"failed to get primary vertices" << std::endl;
      return;
   }

   for (unsigned int i = 0; i < vertices->size(); ++i) {
      const reco::Vertex & vertex = (*vertices)[i];
      /* std::cerr << "Vertex " << i << ":" << std::endl;
         std::cerr << "\tx:" << std::setw(6) << std::setprecision(1) << std::fixed << vertex.x() * 10000. << " um ± " << std::setw(6) << std::setprecision(1) << std::fixed << vertex.xError() * 10000. << " um" << std::endl;
         std::cerr << "\ty:" << std::setw(6) << std::setprecision(1) << std::fixed << vertex.y() * 10000. << " um ± " << std::setw(6) << std::setprecision(1) << std::fixed << vertex.yError() * 10000. << " um" << std::endl;
         std::cerr << "\tz:" << std::setw(6) << std::setprecision(1) << std::fixed << vertex.z()          << " cm ± " << std::setw(6) << std::setprecision(1) << std::fixed << vertex.zError() * 10000. << " um" << std::endl;
         std::cerr << std::endl;
       */
      std::stringstream s;
      s << "Primary Vertex " << i;
      TEveCompound *vList = new TEveCompound(s.str().c_str(),s.str().c_str());
      vList->OpenCompound();
      //guarantees that CloseCompound will be called no matter what happens
      boost::shared_ptr<TEveCompound> sentry(vList,boost::mem_fn(&TEveCompound::CloseCompound));
      gEve->AddElement( vList, list );

      // put point first
      TEvePointSet* pointSet = new TEvePointSet();
      pointSet->SetMainColor(item->defaultDisplayProperties().color());
      pointSet->SetNextPoint( vertex.x(), vertex.y(), vertex.z() );
      vList->AddElement(pointSet);
   }
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(FWVertexRPZProxyBuilder,reco::VertexCollection,"Vertices");
