#include "RVersion.h"
#include "TColor.h"
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveGeoNode.h"
#include "TGeoSphere.h"
#include "TGeoMatrix.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Vertices/interface/VerticesProxy3DBuilder.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <cmath>

static const double scaleError  = 1000.;

VerticesProxy3DBuilder::VerticesProxy3DBuilder() { }

VerticesProxy3DBuilder::~VerticesProxy3DBuilder() { }

void VerticesProxy3DBuilder::build(const FWEventItem* item, TEveElementList** product)
{
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
    std::cout <<"failed to get primary vertices" << std::endl;
    return;
  }
  
  for (unsigned int i = 0; i < vertices->size(); ++i) {
    const reco::Vertex & vertex = (*vertices)[i];

    std::stringstream s;
    s << "Primary Vertex " << i;
    TEveElementList *vList = new TEveElementList(s.str().c_str());
    gEve->AddElement( vList, list );

    // actual 3D shape
    TGeoSphere * sphere = new TGeoSphere(0.0, 1.0);

    // this is just an approximation - the full 3D covariance matrix could be used to show the correct correlations
    TGeoScale       dimension(vertex.xError() * scaleError, vertex.yError() * scaleError, vertex.zError() * scaleError);
    TGeoTranslation position(vertex.x(), vertex.y(), vertex.z());

    // EVE-managed shape
    TEveGeoShape * shape = new TEveGeoShape();
    shape->SetShape(sphere);
    shape->SetTransMatrix(position * dimension);
    shape->SetMainColor(item->defaultDisplayProperties().color());
    shape->SetPickable(kTRUE);
    vList->AddElement(shape);
   }
}
