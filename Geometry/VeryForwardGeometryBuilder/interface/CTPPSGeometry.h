/****************************************************************************
*
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_CTPPSGeometry
#define Geometry_VeryForwardGeometryBuilder_CTPPSGeometry

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"

#include <map>
#include <set>

/**
 * \brief The manager class for TOTEM RP geometry.
 *
 * This is kind of "public relation class" for the tree structure of DetGeomDesc. It provides convenient interface to
 * answer frequently asked questions about the geometry of TOTEM Roman Pots. These questions are of type:\n
 *  - What is the geometry (shift, roatation, material, etc.) of detector with id xxx?\n
 *  - If RP ID is xxx, which are the sensorss inside this pot?\n
 *  - If hit position in local detector coordinate system is xxx, what is the hit position in global c.s.?\n
 * etc. (see the comments in definition bellow)
 **/

class CTPPSGeometry
{
  public:
    typedef std::map<unsigned int, const DetGeomDesc* > mapType;
    typedef std::map<int, const DetGeomDesc* > RPDeviceMapType;
    typedef std::map<unsigned int, std::set<unsigned int> > mapSetType;

    CTPPSGeometry() {}
    ~CTPPSGeometry() {}

    /// build up from DetGeomDesc
    CTPPSGeometry(const DetGeomDesc * gd)
    {
      build(gd);
    }

    /// build up from DetGeomDesc structure, return 0 = success
    char build(const DetGeomDesc *);          

    ///\brief adds an item to the map (detector ID --> DetGeomDesc)
    /// performs necessary checks, returns 0 if succesful
    char addSensor(unsigned int, const DetGeomDesc * &);

    ///\brief adds a RP box to a map
    char addRP(unsigned int id, const DetGeomDesc * &det_geom_desc);


    ///\brief returns geometry of a detector
    /// performs necessary checks, returns NULL if fails
    const DetGeomDesc *getSensor(unsigned int id) const;

    /// returns geometry of a RP box
    const DetGeomDesc *getRP(unsigned int id) const;


    /// begin iterator over sensors
    mapType::const_iterator beginSensor() const
    {
      return theMap.begin();
    }

    /// end iterator over sensors
    mapType::const_iterator endSensor() const
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


    /// coordinate transformations between local<-->global reference frames
    /// dimensions in mm
    /// sensor id expected
    CLHEP::Hep3Vector localToGlobal(const DetGeomDesc *gd, const CLHEP::Hep3Vector& r) const;
    CLHEP::Hep3Vector globalToLocal(const DetGeomDesc *gd, const CLHEP::Hep3Vector& r) const;
    CLHEP::Hep3Vector localToGlobal(unsigned int id, const CLHEP::Hep3Vector& r) const;
    CLHEP::Hep3Vector globalToLocal(unsigned int id, const CLHEP::Hep3Vector& r) const;

    /// direction transformations between local<-->global reference frames
    /// sensor id expected
    CLHEP::Hep3Vector localToGlobalDirection(unsigned int id, const CLHEP::Hep3Vector& dir) const;
    CLHEP::Hep3Vector globalToLocalDirection(unsigned int id, const CLHEP::Hep3Vector& dir) const;


    /// returns translation (position) of a detector
    /// sensor id expected
    CLHEP::Hep3Vector getSensorTranslation(unsigned int id) const;

    /// returns position of a RP box
    /// RP id expected
    CLHEP::Hep3Vector getRPTranslation(unsigned int id) const;


    /// after checks returns set of station ids corresponding to the given arm id
    std::set<unsigned int> const& getStationsInArm(unsigned int) const;

    /// after checks returns set of RP ids corresponding to the given station id
    std::set<unsigned int> const& getRPsInStation(unsigned int) const;
    
    /// after checks returns set of sensor ids corresponding to the given RP id
    std::set<unsigned int> const& getSensorsInRP(unsigned int) const;

  protected:
    /// map: sensor id --> DetGeomDesc
    mapType theMap;
    
    /// map: rp id --> DetGeomDesc
    RPDeviceMapType theRomanPotMap;           

    ///\brief map: parent ID -> set of subelements
    /// E.g. stationsInArm is map of arm ID -> set of stations (in that arm)
    mapSetType stationsInArm, rpsInStation, detsInRP;
};

#endif
