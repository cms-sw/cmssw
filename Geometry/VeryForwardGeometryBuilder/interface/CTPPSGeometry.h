/****************************************************************************
*
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_CTPPSGeometry
#define Geometry_VeryForwardGeometryBuilder_CTPPSGeometry

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSDDDNames.h"

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

class CTPPSGeometry {
public:
  typedef std::map<unsigned int, const DetGeomDesc*> mapType;
  typedef std::map<int, const DetGeomDesc*> RPDeviceMapType;
  typedef std::map<unsigned int, std::set<unsigned int> > mapSetType;

  using Vector = DetGeomDesc::Translation;

  CTPPSGeometry() {}
  ~CTPPSGeometry() {}

  /// build up from DetGeomDesc
  CTPPSGeometry(const DetGeomDesc* gd) { build(gd); }

  /// build up from DetGeomDesc structure
  void build(const DetGeomDesc*);

  //----- setters and getters

  ///\brief adds an item to the map (detector ID --> DetGeomDesc)
  /// performs necessary checks
  /// \return true if successful, false if the sensor is already present
  bool addSensor(unsigned int, const DetGeomDesc*&);

  ///\brief adds a RP box to a map
  /// \return true if successful, false if the RP is already present
  bool addRP(unsigned int id, const DetGeomDesc*&);

  ///\brief returns geometry of a detector
  /// performs necessary checks, returns NULL if fails
  const DetGeomDesc* sensor(unsigned int id) const;
  const DetGeomDesc* sensorNoThrow(unsigned int id) const noexcept;

  /// returns geometry of a RP box
  const DetGeomDesc* rp(unsigned int id) const;
  const DetGeomDesc* rpNoThrow(unsigned int id) const noexcept;

  //----- objects iterators

  /// begin iterator over sensors
  mapType::const_iterator beginSensor() const { return sensors_map_.begin(); }
  /// end iterator over sensors
  mapType::const_iterator endSensor() const { return sensors_map_.end(); }

  /// begin iterator over RPs
  RPDeviceMapType::const_iterator beginRP() const { return rps_map_.begin(); }
  /// end iterator over RPs
  RPDeviceMapType::const_iterator endRP() const { return rps_map_.end(); }

  //----- translators

  /// coordinate transformations between local<-->global reference frames
  /// dimensions in mm
  /// sensor id expected
  Vector localToGlobal(const DetGeomDesc*, const Vector&) const;
  Vector globalToLocal(const DetGeomDesc*, const Vector&) const;
  Vector localToGlobal(unsigned int, const Vector&) const;
  Vector globalToLocal(unsigned int, const Vector&) const;

  /// direction transformations between local and global reference frames
  /// \param id sensor id
  Vector localToGlobalDirection(unsigned int id, const Vector&) const;
  /// direction transformations between global and local reference frames
  /// \param id sensor id
  Vector globalToLocalDirection(unsigned int id, const Vector&) const;

  /// returns translation (position) of a detector
  /// \param id sensor id
  Vector sensorTranslation(unsigned int id) const;

  /// returns position of a RP box
  /// \param id RP id
  Vector rpTranslation(unsigned int id) const;

  /// after checks returns set of station ids corresponding to the given arm id
  const std::set<unsigned int>& stationsInArm(unsigned int) const;

  /// after checks returns set of RP ids corresponding to the given station id
  const std::set<unsigned int>& rpsInStation(unsigned int) const;

  /// after checks returns set of sensor ids corresponding to the given RP id
  const std::set<unsigned int>& sensorsInRP(unsigned int) const;

protected:
  /// map: sensor id --> DetGeomDesc
  mapType sensors_map_;

  /// map: rp id --> DetGeomDesc
  RPDeviceMapType rps_map_;

  ///\brief map: parent ID -> set of subelements
  /// E.g. \a stations_in_arm_ is map of arm ID -> set of stations (in that arm)
  mapSetType stations_in_arm_, rps_in_station_, dets_in_rp_;
};

#endif
