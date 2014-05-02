#ifndef Geometry_CaloTopology_ShashlikTopology_h
#define Geometry_CaloTopology_ShashlikTopology_h 1

/**
 * F.Ratnikov, April 2014
 * Extension of original implementation from S. Banerjee
 */

#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/HGCalCommonData/interface/ShashlikDDDConstants.h"
#include <vector>
#include <iostream>


class ShashlikTopology : public CaloSubdetectorTopology {
  
 public:
  /// create a new Topology
  ShashlikTopology(const ShashlikDDDConstants& sdcons);
  
  /// virtual destructor
  virtual ~ShashlikTopology() { }  
  
  /// move the Topology north (increment iy)  
  virtual DetId  goNorth(const DetId& id) const {
    return incrementIy(EKDetId(id));
  }
  virtual std::vector<DetId> north(const DetId& id) const { 
    EKDetId nextId= goNorth(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==EKDetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  /// move the Topology south (decrement iy)
  virtual DetId goSouth(const DetId& id) const {
    return decrementIy(EKDetId(id));
  }
  virtual std::vector<DetId> south(const DetId& id) const { 
    EKDetId nextId= goSouth(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==EKDetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  /// move the Topology east (positive ix)
  virtual DetId  goEast(const DetId& id) const {
    return incrementIx(EKDetId(id));
  }
  virtual std::vector<DetId> east(const DetId& id) const { 
    EKDetId nextId=goEast(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==EKDetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  /// move the Topology west (negative ix)
  virtual DetId goWest(const DetId& id) const {
    return decrementIx(EKDetId(id));
  }
  virtual std::vector<DetId> west(const DetId& id) const { 
    EKDetId nextId=goWest(id);
    std::vector<DetId> vNeighborsDetId;
    if (! (nextId==EKDetId(0)))
      vNeighborsDetId.push_back(DetId(nextId.rawId()));
    return vNeighborsDetId;
  }
  
  virtual std::vector<DetId> up(const DetId& /*id*/) const {
    std::cout << "ShashlikTopology::up() not yet implemented" << std::endl; 
    std::vector<DetId> vNeighborsDetId;
    return  vNeighborsDetId;
  }
  
  virtual std::vector<DetId> down(const DetId& /*id*/) const {
    std::cout << "ShashlikTopology::down() not yet implemented" << std::endl; 
    std::vector<DetId> vNeighborsDetId;
    return  vNeighborsDetId;
  }
  
  ///Dense indexing for geometry cells
  virtual size_t cellHashSize() const {return 2*smodules_*modules_;}
  virtual size_t cell2denseId(const DetId& id) const;
  virtual DetId denseId2cell(size_t denseId) const;
  
  ///Dense indexing
  virtual size_t detIdHashSize() const {return cellHashSize()*FIB_MAX*RO_MAX;};
  virtual unsigned int detId2denseId(const DetId& id) const;
  virtual DetId denseId2detId(unsigned int denseId) const;
  
  ///Is this a valid cell id
  virtual bool valid(const DetId& id) const;
  virtual bool validXY(int ix, int iy) const;
  bool validHashIndex(uint32_t ix) const {return (ix < kSizeForDenseIndexing);}
  
  ///Convert (X,Y)<->(SM,M)
  std::pair<int,int>  getSMM(int ix, int iy) const {return sdcons_.getSMM (ix, iy);}
  std::pair<int,int>  getXY(int sm, int mod) const {return sdcons_.getXY (sm, mod);}
  
  
  ///Next to boundary
  bool isNextToBoundary(EKDetId id) const;
  bool isNextToDBoundary(EKDetId id) const;
  bool isNextToRingBoundary(EKDetId id) const;
  
  /** returns a new EKDetId offset by nrStepsX and nrStepsY (can be negative),
   * returns EKDetId(0) if invalid */
  DetId offsetBy(const DetId startId, int nrStepsX, int nrStepsY) const;
  DetId switchZSide(const DetId startId) const;
  
  /** Returns the distance along x-axis in module units between two EKDetId
   * @param a det id of first module
   * @param b det id of second module
   * @return distance
   */
  static int distanceX(const EKDetId& a,const EKDetId& b);
  
  /** Returns the distance along y-axis in module units between two EKDetId
   * @param a det id of first module
   * @param b det id of second module
   * @return distance
   */
  static int distanceY(const EKDetId& a,const EKDetId& b); 
  
  /** Maximum possibility of Fiber number (0:FIB_MAX-1)
   */
  static const int FIB_MAX=6;
  
  /** Maximum possibility of Read-Out type (0:RO_MAX-1)
   */
  static const int RO_MAX=3;
  
  /** Access to underlaying constants 
   */
  const ShashlikDDDConstants& dddConstants () const {return sdcons_;}
  
  int totalSuperModules () const {return smodules_;}
  int totalModules () const {return modules_*smodules_;}
  
 private:
  
  /// move the nagivator to larger ix
  EKDetId incrementIx(const EKDetId& id) const ;
  
  /// move the nagivator to smaller ix
  EKDetId decrementIx(const EKDetId& id) const ;
  
  /// move the nagivator to larger iy
  EKDetId incrementIy(const EKDetId& id) const ;
  
  /// move the nagivator to smaller iy
  EKDetId decrementIy(const EKDetId& id) const;
  
  const ShashlikDDDConstants& sdcons_;
  int smodules_, modules_, nRows_, kEKhalf_;
  size_t kSizeForDenseIndexing;
  void init ();
};

#endif
