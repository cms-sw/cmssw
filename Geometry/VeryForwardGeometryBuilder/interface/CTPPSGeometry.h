/****************************************************************************
*
* Authors:
*  Jan Kašpar jan.kaspar@gmail.com)
*  Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_CTPPSGeometry
#define Geometry_VeryForwardGeometryBuilder_CTPPSGeometry

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"

#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"

#include <map>
#include <set>

class DetId;

/**
 * \brief The manager class for CTPPS RP geometry.
 **/

class CTPPSGeometry
{
  public:
    typedef std::map<unsigned int, DetGeomDesc* > mapType;
    typedef std::map<int, DetGeomDesc* > RPDeviceMapType;
    typedef std::map<unsigned int, std::set<unsigned int> > mapSetType;

    CTPPSGeometry() {}
    ~CTPPSGeometry() {}

    /// build up from DetGeomDesc
    CTPPSGeometry(const DetGeomDesc * gd) {}

    /// build up from DetGeomDesc structure, return 0 = success
    virtual char Build(const DetGeomDesc *) = 0;

    ///\brief adds an item to the map (detector ID --> DetGeomDesc)
    /// performs necessary checks, returns 0 if succesful
    char AddDetector(unsigned int, const DetGeomDesc * &);

    ///\brief adds a RP package (primary vacuum) to a map
    char AddRPDevice(unsigned int id, const DetGeomDesc * &det_geom_desc);

    ///\brief returns geometry of a detector
    /// performs necessary checks, returns NULL if fails
    /// input is raw ID
    DetGeomDesc *GetDetector(unsigned int) const;

    DetGeomDesc const *GetDetector(const CTPPSDetId & id) const
    {
      return GetDetector(id.rawId());
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
    DetGeomDesc *GetRPDevice(unsigned int id) const;

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
    std::set<unsigned int> StationsInArm(unsigned int) const;

    /// after checks returns set of RP corresponding to the given station ID
    std::set<unsigned int> RPsInStation(unsigned int) const;
    
    /// after checks returns set of detectors corresponding to the given RP ID
    /// containts decimal detetector IDs
    std::set<unsigned int> DetsInRP(unsigned int) const;

    /// coordinate transformations between local<-->global reference frames
    /// dimensions in mm, raw ID expected
    CLHEP::Hep3Vector LocalToGlobal(DetGeomDesc *gd, const CLHEP::Hep3Vector r) const;
    CLHEP::Hep3Vector GlobalToLocal(DetGeomDesc *gd, const CLHEP::Hep3Vector r) const;
    CLHEP::Hep3Vector LocalToGlobal(unsigned int id, const CLHEP::Hep3Vector r) const;
    CLHEP::Hep3Vector GlobalToLocal(unsigned int id, const CLHEP::Hep3Vector r) const;

    /// direction transformations between local<-->global reference frames
    /// (dimensions in mm), raw ID expected
    CLHEP::Hep3Vector LocalToGlobalDirection(unsigned int id, const CLHEP::Hep3Vector dir) const;
    CLHEP::Hep3Vector GlobalToLocalDirection(unsigned int id, const CLHEP::Hep3Vector dir) const;

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
