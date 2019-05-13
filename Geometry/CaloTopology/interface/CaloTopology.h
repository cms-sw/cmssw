#ifndef GEOMETRY_CALOTOPOLOGY_CALOTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_CALOTOPOLOGY_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include <map>
#include <memory>
#include <vector>

class CaloSubdetectorTopology;

/** \class CaloTopology
      
$Revision: 1.4 $

\author J. Mans and P. Meridiani
*/

class CaloTopology {
public:

  typedef std::map<int, std::unique_ptr<const CaloSubdetectorTopology>> TopMap ;

  CaloTopology();

  ~CaloTopology();
  /// Register a subdetector Topology
  void setSubdetTopology(DetId::Detector det, int subdet, std::unique_ptr<const CaloSubdetectorTopology> geom);
  /// access the subdetector Topology for the given subdetector directly
  const CaloSubdetectorTopology* getSubdetectorTopology(const DetId& id) const;
  /// access the subdetector Topology for the given subdetector directly
  const CaloSubdetectorTopology* getSubdetectorTopology(DetId::Detector det, int subdet) const;
  /// Is this a valid cell id? 
  bool valid(const DetId& id) const;

  /// Get the neighbors of the given cell in east direction
  std::vector<DetId> east(const DetId& id) const;
  /// Get the neighbors of the given cell in west direction
  std::vector<DetId> west(const DetId& id) const;
  /// Get the neighbors of the given cell in north direction
  std::vector<DetId> north(const DetId& id) const;
  /// Get the neighbors of the given cell in south direction
  std::vector<DetId> south(const DetId& id) const;
  /// Get the neighbors of the given cell in up direction (outward)
  std::vector<DetId> up(const DetId& id) const;
  /// Get the neighbors of the given cell in down direction (inward)
  std::vector<DetId> down(const DetId& id) const;
  /// Get the neighbors of the given cell given direction
  std::vector<DetId> getNeighbours(const DetId& id, const CaloDirection& dir) const;
  /// Get the neighbors of the given cell in a window of given size
  std::vector<DetId> getWindow(const DetId& id, const int& northSouthSize, const int& eastWestSize) const;
  /// Get all the neighbors of the given cell
  std::vector<DetId> getAllNeighbours(const DetId& id) const;

private:
  TopMap theTopologies_;
  int makeIndex(DetId::Detector det, int subdet) const;
};



#endif
