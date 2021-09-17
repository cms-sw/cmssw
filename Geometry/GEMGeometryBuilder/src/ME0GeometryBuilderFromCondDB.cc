/** Implementation of the ME0 Geometry Builder from DDD stored in CondDB
 *
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromCondDB.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "DataFormats/Math/interface/GeantUnits.h"

#include <iostream>
#include <algorithm>

using namespace geant_units::operators;

ME0GeometryBuilderFromCondDB::ME0GeometryBuilderFromCondDB() {}

ME0GeometryBuilderFromCondDB::~ME0GeometryBuilderFromCondDB() {}

ME0Geometry* ME0GeometryBuilderFromCondDB::build(const RecoIdealGeometry& rgeo) {
  const std::vector<DetId>& detids(rgeo.detIds());
  ME0Geometry* geometry = new ME0Geometry();

  std::string name;
  std::vector<double>::const_iterator tranStart;
  std::vector<double>::const_iterator shapeStart;
  std::vector<double>::const_iterator rotStart;
  std::vector<std::string>::const_iterator strStart;

  for (unsigned int id = 0; id < detids.size(); ++id) {
    ME0DetId me0id(detids[id]);

    tranStart = rgeo.tranStart(id);
    shapeStart = rgeo.shapeStart(id);
    rotStart = rgeo.rotStart(id);
    strStart = rgeo.strStart(id);
    name = *(strStart);

    Surface::PositionType pos(
        convertMmToCm(*(tranStart)), convertMmToCm(*(tranStart + 1)), convertMmToCm(*(tranStart + 2)));
    Surface::RotationType rot(*(rotStart + 0),
                              *(rotStart + 1),
                              *(rotStart + 2),
                              *(rotStart + 3),
                              *(rotStart + 4),
                              *(rotStart + 5),
                              *(rotStart + 6),
                              *(rotStart + 7),
                              *(rotStart + 8));

    Bounds* bounds = nullptr;
    float be = convertMmToCm(*(shapeStart + 0));
    float te = convertMmToCm(*(shapeStart + 1));
    float ap = convertMmToCm(*(shapeStart + 2));
    float ti = convertMmToCm(*(shapeStart + 3));
    float nstrip = *(shapeStart + 4);
    float npad = *(shapeStart + 5);
    //  TrapezoidalPlaneBounds*
    bounds = new TrapezoidalPlaneBounds(be, te, ap, ti);

    std::vector<float> pars;
    pars.emplace_back(be);  //b/2;
    pars.emplace_back(te);  //B/2;
    pars.emplace_back(ap);  //h/2;
    pars.emplace_back(nstrip);
    pars.emplace_back(npad);

    ME0EtaPartitionSpecs* e_p_specs = new ME0EtaPartitionSpecs(GeomDetEnumerators::ME0, name, pars);

    //Change of axes for the forward
    Basic3DVector<float> newX(1., 0., 0.);
    Basic3DVector<float> newY(0., 0., 1.);
    //      if (tran[2] > 0. )
    newY *= -1;
    Basic3DVector<float> newZ(0., 1., 0.);
    rot.rotateAxes(newX, newY, newZ);

    BoundPlane* bp = new BoundPlane(pos, rot, bounds);
    ReferenceCountingPointer<BoundPlane> surf(bp);
    ME0EtaPartition* mep = new ME0EtaPartition(me0id, surf, e_p_specs);
    geometry->add(mep);
  }
  return geometry;
}
