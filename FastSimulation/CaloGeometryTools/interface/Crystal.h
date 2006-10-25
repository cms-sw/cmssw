#ifndef FastSimulation_GeometryTool_Crystal_h
#define FastSimulation_GeometryTool_Crystal_h

//CLHEP headers
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Geometry/Plane3D.h"

// Unfortunately, GlobalPoints are also needed
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "DataFormats/DetId/interface/DetId.h"

//FAMOS
#include "FastSimulation/CaloGeometryTools/interface/CrystalNeighbour.h"

#include <vector>

class DetId;

class Crystal
{
 public:
  // side numbering
  //  enum CrystalSide{EAST=0,NORTH=1,WEST=2,SOUTH=3,FRONT=4,BACK=5};
  /// Empty constructor 
  Crystal(){;};
  /// constructor from DetId
  Crystal(const DetId&  cell);

  /// 
  void setCorners(const std::vector<GlobalPoint>& vec,const GlobalPoint& pos);

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
  HepPlane3D getFrontPlane() const;
  /// back plane
  HepPlane3D getBackPlane() const;
  /// lateral planes
  HepPlane3D getLateralPlane(unsigned i) const;
  /// generic access
  HepPlane3D getPlane(const CaloDirection& side) const;

  /// lateral directions
  inline const HepVector3D& getLateralEdge(unsigned i) const {return lateraldirection_[i];};

  /// normal exiting vector for the surface
  HepVector3D exitingNormal(const CaloDirection& side) const;

  static unsigned oppositeDirection(unsigned iside);

  /// for debugging. 
  void getDrawingCoordinates(std::vector<float> &x,std::vector<float> &y,std::vector<float> &z) const;

  /// it might be useful to have a number assigned to the crystal
  inline void setNumber(unsigned n) {number_=n;};

  /// get the number of the crystal
  inline unsigned number() const {return number_;};

  /// Direct acces to the iq-th neighbour
  inline CrystalNeighbour & crystalNeighbour(unsigned iq) { return  neighbours_[iq];}
  
  /// get crystal axis
  inline const HepVector3D & getAxis() const { return crystalaxis_;}

  /// set X0back (it depends on the choosen origin, it isn't purely geometrical)
  inline void setX0Back(double val) {X0back_=val;}

  /// get the X0back
  inline double getX0Back() const { return X0back_;}

  ~Crystal(){;};

 private:
  std::vector<HepPoint3D> corners_;
  DetId cellid_;
  bool dummy_;
  int subdetn_;
  HepPoint3D center_;
  HepPoint3D frontcenter_;
  HepPoint3D backcenter_;
  HepVector3D firstedgedirection_;
  HepVector3D fifthedgedirection_;
  HepVector3D crystalaxis_;
  std::vector<HepVector3D> lateraldirection_;
  unsigned number_;
  std::vector<CrystalNeighbour> neighbours_;
  double X0back_;

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
