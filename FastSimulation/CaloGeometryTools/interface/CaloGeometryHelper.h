#ifndef FastSimulation_CaloGeometryTools_CaloGeometryHelper
#define FastSimulation_CaloGeometryTools_CaloGeometryHelper

#include "FastSimulation/CalorimeterProperties/interface/Calorimeter.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "FastSimulation/CaloGeometryTools/interface/BaseCrystal.h"

#include <vector>
#include <array>

class DetId;
class Crystal;

namespace edm { 
  class ParameterSet;
}

class CaloGeometryHelper:public Calorimeter
{

 public:

  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;

  CaloGeometryHelper();
  CaloGeometryHelper(const edm::ParameterSet& fastCalo);
  ~CaloGeometryHelper();


  // more user friendly getClosestCell  
  DetId getClosestCell(const XYZPoint& point, bool ecal, bool central) const;

  // more user friendly getWindow
  void getWindow(const DetId& pivot,int s1,int s2,std::vector<DetId> &) const;
  
  
  double preshowerZPosition(int layer) const
  {
    return (layer==1) ? psLayer1Z_: psLayer2Z_ ; 
  }
  
  // the Crystal constructor
  void buildCrystal(const DetId& id,Crystal&) const;

  void initialize(double bField);  

  // get the <=8 neighbours
  typedef std::array<DetId,8> NeiVect; 
  const  NeiVect& getNeighbours(const DetId& det) const ;

  inline double magneticField() const {return bfield_;}
  
  // temporary. Requires a missing geometry tool 
  bool borderCrossing(const DetId&, const DetId&) const ;

  bool move(DetId& cell, const CaloDirection& dir,bool fast=true) const;

  inline bool preshowerPresent() const {return preshowerPresent_;};

 private:
  void buildNeighbourArray();
  void buildCrystalArray();
  bool simplemove(DetId& cell, const CaloDirection& dir) const;
  bool diagonalmove(DetId& cell, const CaloDirection& dir) const;

 private:
    // Preshower layer positions
  double psLayer1Z_,psLayer2Z_;

  // array of neighbours the hashed index is used for the first vector
  std::vector< NeiVect >  barrelNeighbours_;
  std::vector< NeiVect >  endcapNeighbours_;

  std::vector<BaseCrystal> barrelCrystals_;
  std::vector<BaseCrystal> endcapCrystals_;

  bool neighbourmapcalculated_;
  
  //mag field at 0,0,0
  double bfield_;
  bool preshowerPresent_;
};
#endif
