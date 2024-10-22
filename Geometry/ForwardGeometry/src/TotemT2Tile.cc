/****************************************************************************
*
* This is a part of TOTEM offline software.
* Author:
*   Laurent Forthomme
*
****************************************************************************/

#include "Geometry/ForwardGeometry/interface/TotemT2Tile.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include <TGeoManager.h>

TotemT2Tile::TotemT2Tile() {}

TotemT2Tile::TotemT2Tile(const DetGeomDesc* dgd) {
  centre_ = GlobalPoint{(float)dgd->translation().x(),
                        (float)dgd->translation().y(),
                        (float)dgd->parentZPosition()};  // retrieve the plane position for z coordinate
  double angle = 0.;
  {  // get azimutal component of tile rotation
    AlgebraicMatrix33 mat;
    dgd->rotation().GetRotationMatrix(mat);
    angle = acos(mat[0][0]);
  }
  TGeoCombiTrans place(centre_.x(), centre_.y(), centre_.z(), new TGeoRotation("tile_rot", angle, 0., 0.));
  TGeoManager mgr;
  auto* box = mgr.MakeBox("top", nullptr, 0., 0., 0.);
  mgr.SetTopVolume(box);
  auto* vol = mgr.MakeTrd1("tile", nullptr, dgd->params()[4], dgd->params()[8], dgd->params()[3], dgd->params()[0]);
  box->AddNode(vol, 1, &place);
}

TotemT2Tile::~TotemT2Tile() {}
