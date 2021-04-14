#ifndef Geometry_TrackerGeometryBuilder_PixelTopologyMap_H
#define Geometry_TrackerGeometryBuilder_PixelTopologyMap_H

// system include files
#include <map>
#include <memory>
#include <iostream>
#include <iomanip>  // std::setw

// user include files
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "TrackerGeometry.h"

/**
 * A specific Pixel Tracker class to determine the number of ladders / modules in PXB
   and number of blades/modules in PXF
 */
class PixelTopologyMap {
public:
  PixelTopologyMap(const TrackerGeometry* geom, const TrackerTopology* topo)
      : m_trackerTopo{topo}, m_trackerGeom{geom} {
    // build the maps
    buildTopologyMaps();
  }

  ~PixelTopologyMap() = default;

  // getter methods

  inline const unsigned getPXBLadders(unsigned int lay) const { return m_pxbMap.at(lay).first; }
  inline const unsigned getPXBModules(unsigned int lay) const { return m_pxbMap.at(lay).second; }
  inline const unsigned getPXFBlades(int disk) const { return m_pxfMap.at(std::abs(disk)).first; }
  inline const unsigned getPXFModules(int disk) const { return m_pxfMap.at(std::abs(disk)).second; }

  // printout
  void printAll(std::ostream& os) const;

private:
  void buildTopologyMaps();

  const TrackerTopology* m_trackerTopo;
  const TrackerGeometry* m_trackerGeom;

  std::map<unsigned, std::pair<unsigned, unsigned>> m_pxbMap;
  std::map<unsigned, std::pair<unsigned, unsigned>> m_pxfMap;
};

inline std::ostream& operator<<(std::ostream& os, PixelTopologyMap map) {
  std::stringstream ss;
  map.printAll(ss);
  os << ss.str();
  return os;
}

#endif
