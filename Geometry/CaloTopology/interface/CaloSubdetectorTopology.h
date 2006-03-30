#ifndef TOPOLOGY_CALOTOPOLOGY_CALOSUBDETECTORTOPOLOGY_H
#define TOPOLOGY_CALOTOPOLOGY_CALOSUBDETECTORTOPOLOGY_H 1

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

/** \class CaloSubdetectorTopology
      
$Date: $
$Revision: $
\author P.Meridiani INFN Roma1
\author J. Mans - Minnesota
*/

class CaloSubdetectorTopology {
public:
  /// standard constructor
  CaloSubdetectorTopology() {};
  /// virtual destructor
  virtual ~CaloSubdetectorTopology() { }
  /// is this detid present in the Topology?
  virtual bool valid(const DetId& id) const { return false; };
  /** Get the neighbors of the given cell in east direction*/
  virtual std::vector<DetId> east(const DetId& id) const = 0;
  /** Get the neighbors of the given cell in west direction*/
  virtual std::vector<DetId> west(const DetId& id) const = 0;
  /** Get the neighbors of the given cell in north direction*/
  virtual std::vector<DetId> north(const DetId& id) const = 0;
  /** Get the neighbors of the given cell in south direction*/
  virtual std::vector<DetId> south(const DetId& id) const = 0;
  /** Get the neighbors of the given cell in up direction (outward)*/
  virtual std::vector<DetId> up(const DetId& id) const = 0;
  /** Get the neighbors of the given cell in down direction (inward)*/
  virtual std::vector<DetId> down(const DetId& id) const = 0;
};


#endif
