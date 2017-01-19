/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DDDTotemRPCommon.h"
#include <iostream>
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

char
TotemRPGeometry::Build(const DetGeomDesc *gD)
{
  // propagate through the GeometricalDet structure and add
  // all detectors to 'theMap'
  deque<const DetGeomDesc *> buffer;
  buffer.push_back(gD);
  while (buffer.size() > 0)
  {
    const DetGeomDesc *d = buffer.front();
    buffer.pop_front();

    // check if it is RP detector
    if (! d->name().name().compare(DDD_TOTEM_RP_DETECTOR_NAME))
      AddDetector(d->geographicalID(), d);

    // check if it is RP device (primary vacuum)
    if (! d->name().name().compare(DDD_TOTEM_RP_PRIMARY_VACUUM_NAME))
      AddRPDevice(d->geographicalID(), d);
    
    for (unsigned int i = 0; i < d->components().size(); i++)
      buffer.push_back(d->components()[i]);
  }

  // build sets from theMap
  BuildSets();

  return 0;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
TotemRPGeometry::GetDetEdgePosition(unsigned int id) const
{
    // hardcoded for now, values taken from RP_Hybrid.xml
    // +-------+
    // |       |
    // |   + (0,0)
    //  *(x,y) |
    //   \-----+
    // x=-RP_Det_Size_a/2+RP_Det_Edge_Length/(2*sqrt(2))
    // y=x
    // ideally we would get this from the geometry in the event setup
    const double x=-36.07/2+22.276/(2*sqrt(2));
    return LocalToGlobal(id, CLHEP::Hep3Vector(x, x, 0.));
}

// Left edge: -18.0325, -2.2209 ; Right edge: -2.2209, -18.0325

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
TotemRPGeometry::GetDetEdgeNormalVector(unsigned int id) const
{
    return CTPPSGeometry::GetDetector(id)->rotation() * CLHEP::Hep3Vector(-sqrt(2)/2, -sqrt(2)/2, 0.);
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
TotemRPGeometry::GetRPThinFoilPosition(int copy_no) const
{
	// hardcoded for now, taken from RP_Box.xml:RP_Box_primary_vacuum_y
	// ideally we would get this from the geometry in the event setup
	return LocalToGlobal(GetRPDevice(copy_no), CLHEP::Hep3Vector(0., -135.65/2.0, 0.));
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
TotemRPGeometry::GetRPThinFoilNormalVector(int copy_no) const
{
	return GetRPDevice(copy_no)->rotation() * CLHEP::Hep3Vector(0., -1., 0.);
}

