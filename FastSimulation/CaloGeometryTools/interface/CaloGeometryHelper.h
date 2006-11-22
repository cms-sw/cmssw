#ifndef FastSimulation_CaloGeometryTools_CaloGeometryHelper
#define FastSimulation_CaloGeometryTools_CaloGeometryHelper

#include "FastSimulation/CalorimeterProperties/interface/Calorimeter.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CLHEP/Geometry/Point3D.h"

#include <vector>

class DetId;
class Crystal;

class CaloGeometryHelper:public Calorimeter
{
 public:
  CaloGeometryHelper();
  CaloGeometryHelper(const edm::ParameterSet& fastCalo);
  ~CaloGeometryHelper();


  // more user friendly getClosestCell  
  DetId getClosestCell(const HepPoint3D& point, bool ecal, bool central) const;

  // more user friendly getWindow
  void getWindow(const DetId& pivot,int s1,int s2,std::vector<DetId> &) const;
  
  
  double preshowerZPosition(int layer) const
  {
    return (layer==1) ? psLayer1Z_: psLayer2Z_ ; 
  }
  
  // the Crystal constructor
  void buildCrystal(const DetId& id,Crystal&) const;

  void initialize();  

  // get the <=8 neighbours
  const std::vector<DetId> & getNeighbours(const DetId& det) const ;

  inline double magneticField() const {return bfield_;}
  
  // temporary. Requires a missing geometry tool 
  bool borderCrossing(const DetId&, const DetId&) const { return false; }

  bool move(DetId& cell, const CaloDirection& dir,bool fast=true) const;

 private:
  void buildNeighbourArray();
  bool simplemove(DetId& cell, const CaloDirection& dir) const;
  bool diagonalmove(DetId& cell, const CaloDirection& dir) const;

 private:
    // Preshower layer positions
  double psLayer1Z_,psLayer2Z_;

  // array of neighbours the hashed index is used for the first vector
  std::vector<std::vector<DetId> > barrelNeighbours_;
  std::vector<std::vector<DetId> > endcapNeighbours_;
  bool neighbourmapcalculated_;
  
  //mag field at 0,0,0
  double bfield_;
};
#endif
