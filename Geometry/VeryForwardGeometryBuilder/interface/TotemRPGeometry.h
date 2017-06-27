/****************************************************************************
*
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_TotemRPGeometry
#define Geometry_VeryForwardGeometryBuilder_TotemRPGeometry

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"

#include <map>
#include <set>

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

class TotemRPGeometry
{
  public:
    typedef std::map<unsigned int, const DetGeomDesc* > mapType;
    typedef std::map<int, const DetGeomDesc* > RPDeviceMapType;
    typedef std::map<unsigned int, std::set<unsigned int> > mapSetType;

    TotemRPGeometry() {}
    ~TotemRPGeometry() {}

    /// build up from DetGeomDesc
    TotemRPGeometry(const DetGeomDesc * gd)
    {
      Build(gd);
    }

    /// build up from DetGeomDesc structure, return 0 = success
    char Build(const DetGeomDesc *);          

    ///\brief adds an item to the map (detector ID --> DetGeomDesc)
    /// performs necessary checks, returns 0 if succesful
    char AddDetector(unsigned int, const DetGeomDesc * &);

    ///\brief adds a RP package (primary vacuum) to a map
    char AddRPDevice(unsigned int id, const DetGeomDesc * &det_geom_desc);

    ///\brief returns geometry of a detector
    /// performs necessary checks, returns NULL if fails
    /// input is raw ID
    const DetGeomDesc *GetDetector(unsigned int) const;

    const DetGeomDesc* GetDetector( const TotemRPDetId& id ) const
    {
      return GetDetector( id.rawId() );
    }
    const DetGeomDesc* GetDetector( const CTPPSDetId& id ) const
    {
      return GetDetector( id.rawId() );
    }

    /// same as GetDetector
    DetGeomDesc const *operator[] (unsigned int id) const
    {
      return GetDetector(id);
    }

    /// returns the position of the edge of a detector
    CLHEP::Hep3Vector GetDetEdgePosition(unsigned int id) const;

    /// returns a normal vector for the edge of a detector
    CLHEP::Hep3Vector GetDetEdgeNormalVector(unsigned int id) const;

    /// returns geometry of a RP box
    const DetGeomDesc *GetRPDevice(unsigned int id) const;

    /// returns the (outer) position of the thin foil of a RP box
    CLHEP::Hep3Vector GetRPThinFoilPosition(int copy_no) const;

    /// returns a normal vector for the thin foil of a RP box
    CLHEP::Hep3Vector GetRPThinFoilNormalVector(int copy_no) const;

    /// begin iterator over (silicon) detectors
    mapType::const_iterator beginDet() const
    {
      return theMap.begin();
    }

    /// end iterator over (silicon) detectors
    mapType::const_iterator endDet() const
    {
      return theMap.end();
    }

    /// begin iterator over RPs
    RPDeviceMapType::const_iterator beginRP() const
    {
      return theRomanPotMap.begin();
    }

    /// end iterator over RPs
    RPDeviceMapType::const_iterator endRP() const
    {
      return theRomanPotMap.end();
    }

    ///\brief builds maps element ID --> set of subelements
    /// (re)builds stationsInArm, rpsInStation, detsInRP out of theMap
    void BuildSets();

    /// after checks returns set of stations corresponding to the given arm ID
    std::set<unsigned int> const& StationsInArm(unsigned int) const;

    /// after checks returns set of RP corresponding to the given station ID
    std::set<unsigned int> const& RPsInStation(unsigned int) const;
    
    /// after checks returns set of detectors corresponding to the given RP ID
    /// containts decimal detetector IDs
    std::set<unsigned int> const& DetsInRP(unsigned int) const;

    /// coordinate transformations between local<-->global reference frames
    /// dimensions in mm, raw ID expected
    CLHEP::Hep3Vector LocalToGlobal(const DetGeomDesc *gd, const CLHEP::Hep3Vector& r) const;
    CLHEP::Hep3Vector GlobalToLocal(const DetGeomDesc *gd, const CLHEP::Hep3Vector& r) const;
    CLHEP::Hep3Vector LocalToGlobal(unsigned int id, const CLHEP::Hep3Vector& r) const;
    CLHEP::Hep3Vector GlobalToLocal(unsigned int id, const CLHEP::Hep3Vector& r) const;

    /// direction transformations between local<-->global reference frames
    /// (dimensions in mm), raw ID expected
    CLHEP::Hep3Vector LocalToGlobalDirection(unsigned int id, const CLHEP::Hep3Vector& dir) const;
    CLHEP::Hep3Vector GlobalToLocalDirection(unsigned int id, const CLHEP::Hep3Vector& dir) const;

    /// returns translation (position) of a detector
    /// raw ID as input
    CLHEP::Hep3Vector GetDetTranslation(unsigned int id) const;

    /// returns (the transverse part of) the readout direction in global coordinates
    /// raw ID expected
    void GetReadoutDirection(unsigned int id, double &dx, double &dy) const;

    /// position of a RP package (translation z corresponds to the first plane - TODO check it)
    CLHEP::Hep3Vector GetRPGlobalTranslation(int copy_no) const;
    CLHEP::HepRotation GetRPGlobalRotation(int copy_no) const;

    ///< returns number of detectors in the geometry (size of theMap)
    unsigned int NumberOfDetsIncluded() const
    {
      return theMap.size();
    }

  protected:
    mapType theMap;                           ///< map: detectorID --> DetGeomDesc
    RPDeviceMapType theRomanPotMap;           ///< map: RPID --> DetGeomDesc

    ///\brief map: parent ID -> set of subelements
    /// E.g. stationsInArm is map of arm ID -> set of stations (in that arm)
    mapSetType stationsInArm, rpsInStation, detsInRP;
};

#endif
