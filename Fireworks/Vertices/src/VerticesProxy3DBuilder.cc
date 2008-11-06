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
#include "TEveCompound.h"
#include "TEvePointSet.h"

#include <cmath>

static const double scaleError  = 1.;

VerticesProxy3DBuilder::VerticesProxy3DBuilder() { }

VerticesProxy3DBuilder::~VerticesProxy3DBuilder() { }

void VerticesProxy3DBuilder::build(const FWEventItem* item, TEveElementList** product)
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
  
  // actual 3D shape
  TGeoSphere * sphere = new TGeoSphere(0.0, 1.0);

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


    // this is just an approximation - the full 3D covariance matrix could be used to show the correct correlations
    TGeoScale dimension(
        (vertex.xError() ? vertex.xError() : 0.0015) * scaleError,      // use a default error of 15 um
        (vertex.yError() ? vertex.yError() : 0.0015) * scaleError,      // use a default error of 15 um
        (vertex.zError() ? vertex.zError() : 5.0000) * scaleError);     // use a default error of  5 cm
    TGeoTranslation position(vertex.x(), vertex.y(), vertex.z());

    // EVE-managed shape
    TEveGeoShape * shape = new TEveGeoShape("vertices");
    shape->SetShape(sphere);
    shape->SetTransMatrix(position * dimension);
    shape->SetMainColor(item->defaultDisplayProperties().color());
    shape->SetPickable(kTRUE);
    vList->AddElement(shape);
   }
}

REGISTER_FWRPZDATAPROXYBUILDER(VerticesProxy3DBuilder,reco::VertexCollection,"Vertices");
