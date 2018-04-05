#ifndef FastSimulation_GeometryTool_BaseCrystal_h
#define FastSimulation_GeometryTool_BaseCrystal_h

//Data Formats
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/Plane3D.h"

// Unfortunately, GlobalPoints are also needed
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloDirectionOperations.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>

class DetId;

class BaseCrystal
{

 public:

  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;
  typedef ROOT::Math::Plane3D Plane3D;

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
  void setCorners(const CaloCellGeometry::CornersVec& vec,const GlobalPoint& pos);

  // inline const std::vector<XYZPoint>& getCorners() const {return corners_;}

  /// get the i-th corner
  inline const XYZPoint& getCorner(unsigned i) const { return corners_[i];};
  /// get 1/8*(Sum of corners)
  inline const XYZPoint& getCenter() const {return center_;};
  /// get front center
  inline const XYZPoint& getFrontCenter() const {return frontcenter_;};
  /// get front center
  inline const XYZPoint & getBackCenter() const {return backcenter_;}
  /// Direction of the first edge 
  inline const XYZVector& getFirstEdge() const {return  firstedgedirection_;}
  /// Direction of the fifth edge 
  inline const XYZVector& getFifthEdge() const {return  fifthedgedirection_;}
  /// get the DetId
  inline const DetId & getDetId() const {return cellid_;};
  /// get the subdector
  inline const int getSubdetNumber() const {return subdetn_;}

  /// get the lateral edges
  void getLateralEdges(unsigned i,XYZPoint&,XYZPoint&)const;
  /// coordinates of the front side
  void getFrontSide(XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const;
  void getFrontSide(std::vector<XYZPoint>& corners) const;
  /// Coordinates of the back side
  void getBackSide(XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const;
  void getBackSide(std::vector<XYZPoint>& corners) const;
  /// Coordinates of the i=th lateral side
  void getLateralSide(unsigned i,XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const;
  void getLateralSide(unsigned i,std::vector<XYZPoint>& corners) const;
  /// generic access
  void getSide(const CaloDirection& side, XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const;
  void getSide(const CaloDirection& side, std::vector<XYZPoint>& corners) const;



  /// front plane
  inline const Plane3D& getFrontPlane() const {return lateralPlane_[4];}
  /// back plane
  inline const  Plane3D& getBackPlane() const {return lateralPlane_[5];}
  /// lateral planes
  inline const Plane3D& getLateralPlane(unsigned i) const {return lateralPlane_[i];};
  /// generic access
  const Plane3D& getPlane(const CaloDirection& side) const {return lateralPlane_[CaloDirectionOperations::Side(side)];}

  /// lateral directions
  inline const XYZVector& getLateralEdge(unsigned i) const {return lateraldirection_[i];};

  /// normal exiting vector for the surface
  inline const XYZVector& exitingNormal(const CaloDirection& side) const {return exitingNormal_[CaloDirectionOperations::Side(side)];};

  static unsigned oppositeDirection(unsigned iside);

  /// for debugging. 
  void getDrawingCoordinates(std::vector<float> &x,std::vector<float> &y,std::vector<float> &z) const;

  /// get crystal axis
  inline const XYZVector & getAxis() const { return crystalaxis_;}

  void print() const;

 private:
  void computeBasicProperties();

 private:
  XYZPoint corners_[8];
  DetId cellid_;
  int subdetn_;
  XYZPoint center_;
  XYZPoint frontcenter_;
  XYZPoint backcenter_;
  XYZVector firstedgedirection_;
  XYZVector fifthedgedirection_;
  XYZVector crystalaxis_;
  XYZVector lateraldirection_[4];
  Plane3D lateralPlane_[6];
  XYZVector exitingNormal_[6];
};
#endif
