#ifndef FastSimulation_GeometryTool_Crystal_h
#define FastSimulation_GeometryTool_Crystal_h

// Data Formats
#include "Math/GenVector/Plane3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

// Unfortunately, GlobalPoints are also needed
#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "DataFormats/DetId/interface/DetId.h"

//FAMOS
#include "FastSimulation/CaloGeometryTools/interface/CrystalNeighbour.h"
#include "FastSimulation/CaloGeometryTools/interface/BaseCrystal.h"

#include <vector>

class DetId;

class Crystal
{

 public:

  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;
  typedef ROOT::Math::Plane3D Plane3D;

  // side numbering
  //  enum CrystalSide{EAST=0,NORTH=1,WEST=2,SOUTH=3,FRONT=4,BACK=5};
  /// Empty constructor 
  Crystal(){number_ = 0;};
  /// constructor from DetId
  Crystal(const DetId&  cell,const BaseCrystal* bc=0);

  /// get the i-th corner
  inline const XYZPoint& getCorner(unsigned i) const { return myCrystal_->getCorner(i);};
  /// get 1/8*(Sum of corners)
  inline const XYZPoint& getCenter() const {return myCrystal_->getCenter();};
  /// get front center
  inline const XYZPoint& getFrontCenter() const {return myCrystal_->getFrontCenter();};
  /// get front center
  inline const XYZPoint & getBackCenter() const {return myCrystal_->getBackCenter();}
  /// Direction of the first edge 
  inline const XYZVector& getFirstEdge() const {return  myCrystal_->getFirstEdge();}
  /// Direction of the fifth edge 
  inline const XYZVector& getFifthEdge() const {return myCrystal_->getFifthEdge();}
  /// get the DetId
  inline const DetId & getDetId() const {return cellid_;};
  /// get the subdector
  inline const int getSubdetNumber() const {return myCrystal_->getSubdetNumber();}
  void print() const {return myCrystal_->print();}
  /// get the lateral edges
  void getLateralEdges(unsigned i,XYZPoint& a,XYZPoint& b) const {myCrystal_->getLateralEdges(i,a,b); };
  /// coordinates of the front side
  void getFrontSide(XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const {myCrystal_->getFrontSide(a,b,c,d); }
  void getFrontSide(std::vector<XYZPoint>& corners) const {myCrystal_->getFrontSide(corners);}
  /// Coordinates of the back side
  void getBackSide(XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const {myCrystal_->getBackSide(a,b,c,d);}
  void getBackSide(std::vector<XYZPoint>& corners) const {myCrystal_->getBackSide(corners);}
  /// Coordinates of the i=th lateral side
  void getLateralSide(unsigned i,XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const {myCrystal_->getLateralSide(i,a,b,c,d);}
  void getLateralSide(unsigned i,std::vector<XYZPoint>& corners) const {myCrystal_->getLateralSide(i,corners);}
  /// generic access
  void getSide(const CaloDirection& side, XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const
  {myCrystal_->getSide(side,a,b,c,d);}
  void getSide(const CaloDirection& side, std::vector<XYZPoint>& corners) const
  {myCrystal_->getSide(side,corners);}


  /// front plane
  const Plane3D& getFrontPlane() const {return myCrystal_->getFrontPlane();}
  /// back plane
  const Plane3D& getBackPlane() const {return myCrystal_->getBackPlane();}
  /// lateral planes
  const Plane3D& getLateralPlane(unsigned i) const {return myCrystal_->getLateralPlane(i);}
  /// generic access
  const Plane3D& getPlane(const CaloDirection& side) const { return myCrystal_->getPlane(side);}

  /// lateral directions
 inline const XYZVector& getLateralEdge(unsigned i) const {return myCrystal_->getLateralEdge(i);}

  /// normal exiting vector for the surface
 inline const XYZVector& exitingNormal(const CaloDirection& side) const {return myCrystal_->exitingNormal(side);}

  static unsigned oppositeDirection(unsigned iside);

  /// for debugging. 
  void getDrawingCoordinates(std::vector<float> &x,std::vector<float> &y,std::vector<float> &z) const
  {
    myCrystal_->getDrawingCoordinates(x,y,z);
  }

  /// it might be useful to have a number assigned to the crystal
  inline void setNumber(unsigned n) {number_=n;};

  /// get the number of the crystal
  inline unsigned number() const {
    return number_;};

  /// Direct acces to the iq-th neighbour
  inline CrystalNeighbour & crystalNeighbour(unsigned iq) { return  neighbours_[iq];}
  
  /// get crystal axis
  inline const XYZVector & getAxis() const { return myCrystal_->getAxis();}

  /// set X0back (it depends on the choosen origin, it isn't purely geometrical)
  inline void setX0Back(double val) {X0back_=val;}

  /// get the X0back
  inline double getX0Back() const { return X0back_;}

  ~Crystal(){;};

 private:
  unsigned number_;
  DetId cellid_ ;
  std::vector<CrystalNeighbour> neighbours_;
  double X0back_;
  const BaseCrystal * myCrystal_;

 public:
  class crystalEqual
    {
    public:
      crystalEqual(const DetId & cell):ref_(cell)
	{;};
      ~crystalEqual(){;};
      inline bool operator() (const Crystal& xtal) const
	{
	  return (ref_==xtal.getDetId());
	}
    private:
      const DetId& ref_;
    };
};
#endif
