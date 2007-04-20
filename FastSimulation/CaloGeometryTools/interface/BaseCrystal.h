#ifndef FastSimulation_GeometryTool_BaseCrystal_h
#define FastSimulation_GeometryTool_BaseCrystal_h

//CLHEP headers
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Geometry/Plane3D.h"

// Unfortunately, GlobalPoints are also needed
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloDirectionOperations.h"
#include "DataFormats/DetId/interface/DetId.h"

//FAMOS
#include "FastSimulation/CaloGeometryTools/interface/CrystalNeighbour.h"

#include <vector>
#include <iostream>

class DetId;

class BaseCrystal
{
 public:
  // side numbering
  //  enum CrystalSide{EAST=0,NORTH=1,WEST=2,SOUTH=3,FRONT=4,BACK=5};
  /// Empty constructor 
   BaseCrystal(){;};
  /// constructor from DetId
  BaseCrystal(const DetId&  cell);

  /// Copy constructor
     //    BaseCrystal (const BaseCrystal& bc):corners_(bc.getCorners()),cellid_(bc.getDetId())
    //    { 
      //      std::cout << " Copy constructor " ;
    //      computeBasicProperties();
      //      std::cout << " done " << std::endl;
    //    }
  ~BaseCrystal() {;}
  /// 
  void setCorners(const std::vector<GlobalPoint>& vec,const GlobalPoint& pos);

  inline const std::vector<HepPoint3D>& getCorners() const {return corners_;}

  /// get the i-th corner
  inline const HepPoint3D& getCorner(unsigned i) const { return corners_[i];};
  /// get 1/8*(Sum of corners)
  inline const HepPoint3D& getCenter() const {return center_;};
  /// get front center
  inline const HepPoint3D& getFrontCenter() const {return frontcenter_;};
  /// get front center
  inline const HepPoint3D & getBackCenter() const {return backcenter_;}
  /// Direction of the first edge 
  inline const HepVector3D& getFirstEdge() const {return  firstedgedirection_;}
  /// Direction of the fifth edge 
  inline const HepVector3D& getFifthEdge() const {return  fifthedgedirection_;}
  /// get the DetId
  inline const DetId & getDetId() const {return cellid_;};
  /// get the subdector
  inline const int getSubdetNumber() const {return subdetn_;}

  /// get the lateral edges
  void getLateralEdges(unsigned i,HepPoint3D&,HepPoint3D&)const;
  /// coordinates of the front side
  void getFrontSide(HepPoint3D &a,HepPoint3D &b,HepPoint3D &c,HepPoint3D &d) const;
  void getFrontSide(std::vector<HepPoint3D>& corners) const;
  /// Coordinates of the back side
  void getBackSide(HepPoint3D &a,HepPoint3D &b,HepPoint3D &c,HepPoint3D &d) const;
  void getBackSide(std::vector<HepPoint3D>& corners) const;
  /// Coordinates of the i=th lateral side
  void getLateralSide(unsigned i,HepPoint3D &a,HepPoint3D &b,HepPoint3D &c,HepPoint3D &d) const;
  void getLateralSide(unsigned i,std::vector<HepPoint3D>& corners) const;
  /// generic access
  void getSide(const CaloDirection& side, HepPoint3D &a,HepPoint3D &b,HepPoint3D &c,HepPoint3D &d) const;
  void getSide(const CaloDirection& side, std::vector<HepPoint3D>& corners) const;



  /// front plane
  inline const HepPlane3D& getFrontPlane() const {return lateralPlane_[4];}
  /// back plane
  inline const  HepPlane3D& getBackPlane() const {return lateralPlane_[5];}
  /// lateral planes
  inline const HepPlane3D& getLateralPlane(unsigned i) const {return lateralPlane_[i];};
  /// generic access
  const HepPlane3D& getPlane(const CaloDirection& side) const {return lateralPlane_[CaloDirectionOperations::Side(side)];}

  /// lateral directions
  inline const HepVector3D& getLateralEdge(unsigned i) const {return lateraldirection_[i];};

  /// normal exiting vector for the surface
  inline const HepVector3D& exitingNormal(const CaloDirection& side) const {return exitingNormal_[CaloDirectionOperations::Side(side)];};

  static unsigned oppositeDirection(unsigned iside);

  /// for debugging. 
  void getDrawingCoordinates(std::vector<float> &x,std::vector<float> &y,std::vector<float> &z) const;

  /// get crystal axis
  inline const HepVector3D & getAxis() const { return crystalaxis_;}

  void print() const;

 private:
  void computeBasicProperties();

 private:
  std::vector<HepPoint3D> corners_;
  DetId cellid_;
  int subdetn_;
  HepPoint3D center_;
  HepPoint3D frontcenter_;
  HepPoint3D backcenter_;
  HepVector3D firstedgedirection_;
  HepVector3D fifthedgedirection_;
  HepVector3D crystalaxis_;
  std::vector<HepVector3D> lateraldirection_;
  std::vector<HepPlane3D> lateralPlane_;
  std::vector<HepVector3D> exitingNormal_;
};
#endif
