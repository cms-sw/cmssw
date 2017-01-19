/****************************************************************************
*
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_TotemRPGeometry
#define Geometry_VeryForwardGeometryBuilder_TotemRPGeometry

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

class DetId;

/**
 * \ingroup TotemRPGeometry
 * \brief The manager class for TOTEM RP geometry.
 *
 * See schema of \ref TotemRPGeometry "TOTEM RP geometry classes"
 *
 * This is kind of "public relation class" for the tree structure of DetGeomDesc. It provides convenient interface to
 * answer frequently asked questions about the geometry of TOTEM Roman Pots. These questions are of type:\n
 * a) If detector ID is xxx, what is the ID of corresponding station?\n
 * b) What is the geometry (shift, roatation, material, etc.) of detector with id xxx?\n
 * c) If RP ID is xxx, which are the detector IDs inside this pot?\n
 * d) If hit position in local detector coordinate system is xxx, what is the hit position in global c.s.?\n
 * etc. (see the comments in definition bellow)\n
 * This class is built for both ideal and real geometry. I.e. it is produced by TotemRPIdealGeometryESModule in
 * IdealGeometryRecord and similarly for the real geometry
 **/

class TotemRPGeometry : public CTPPSGeometry
{
  public:
    TotemRPGeometry() : CTPPSGeometry() {}
    ~TotemRPGeometry() {}

    /// build up from DetGeomDesc
    TotemRPGeometry(const DetGeomDesc * gd) : CTPPSGeometry( gd )
    {
      Build(gd);
    }

    /// build up from DetGeomDesc structure, return 0 = success
    char Build(const DetGeomDesc *);          

    DetGeomDesc const *GetDetector(const TotemRPDetId & id) const
    {
      return CTPPSGeometry::GetDetector(id.rawId());
    }

    /// returns the position of the edge of a detector
    CLHEP::Hep3Vector GetDetEdgePosition(unsigned int id) const;

    /// returns a normal vector for the edge of a detector
    CLHEP::Hep3Vector GetDetEdgeNormalVector(unsigned int id) const;

    /// returns the (outer) position of the thin foil of a RP box
    CLHEP::Hep3Vector GetRPThinFoilPosition(int copy_no) const;

    /// returns a normal vector for the thin foil of a RP box
    CLHEP::Hep3Vector GetRPThinFoilNormalVector(int copy_no) const;

};

#endif
