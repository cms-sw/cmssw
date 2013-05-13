#ifndef GEOMETRY_CALOTOPOLOGY_CALOTOWERTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_CALOTOWERTOPOLOGY_H 1

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

/** \class CaloTowerTopology
  *  
  * $Date: 2006/08/29 12:33:05 $
  * $Revision: 1.4 $
  * \author J. Mans - Minnesota
  */
class CaloTowerTopology GCC11_FINAL : public CaloSubdetectorTopology {
public:
  /// standard constructor
  CaloTowerTopology() {}
  /// virtual destructor
  virtual ~CaloTowerTopology() { }
  /// is this detid present in the Topology?
  virtual bool valid(const DetId& id) const;
  /** Get the neighbors of the given cell in east direction*/
  virtual std::vector<DetId> east(const DetId& id) const;
  /** Get the neighbors of the given cell in west direction*/
  virtual std::vector<DetId> west(const DetId& id) const;
  /** Get the neighbors of the given cell in north direction*/
  virtual std::vector<DetId> north(const DetId& id) const;
  /** Get the neighbors of the given cell in south direction*/
  virtual std::vector<DetId> south(const DetId& id) const;
  /** Get the neighbors of the given cell in up direction (outward)*/
  virtual std::vector<DetId> up(const DetId& id) const;
  /** Get the neighbors of the given cell in down direction (inward)*/
  virtual std::vector<DetId> down(const DetId& id) const;
};
#endif
