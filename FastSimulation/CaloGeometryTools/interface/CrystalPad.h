#ifndef FastSimulation_CaloGeometryTools_CrystalPad
#define FastSimulation_CaloGeometryTools_CrystalPad

#include "Geometry/CaloTopology/interface/CaloDirection.h"

#include "CLHEP/Vector/TwoVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
//#include "Math/GenVector/Transform3D.h"
#include "FastSimulation/CaloGeometryTools/interface/Transform3DPJ.h"
//C++ headers
#include <vector>


class CrystalPad
{
 public:

  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;
  typedef ROOT::Math::Transform3DPJ Transform3D;
  typedef ROOT::Math::Transform3DPJ::Point Point;


  CrystalPad() { dummy_ = true;};
  /// Order matters. 1234 2341 3412 4123 are ok but not 1324 ....
  CrystalPad(unsigned number, const std::vector<CLHEP::Hep2Vector>& corners);
  /// Constructor from space points, with the description of the local 
  /// frame (origin,vec1,vec2) where vec1 is normal to the plane and vec2 
  /// in the plane
  CrystalPad(unsigned number, int onEcal, 
	     const std::vector<XYZPoint>& corners, 
	     const XYZPoint& origin, 
	     const XYZVector& vec1, 
	     const XYZVector& vec2);

  CrystalPad(unsigned number, 
	     const std::vector<XYZPoint>& corners,
	     const Transform3D&,
	     double scaf=1.,bool bothdirections=false);

  CrystalPad(const CrystalPad& right);

  CrystalPad& operator = (const CrystalPad& rhs );

  ~CrystalPad(){;};
  
  /// Check that the point (in the local frame) is inside the crystal. 
  bool inside(const CLHEP::Hep2Vector & point,bool debug=false) const;
  /// Check that the point (in the global frame) is inside the crystal. 
  //  bool globalinside(XYZPoint) const;

  /// coordinates of the point in the local frame
  //  CLHEP::Hep2Vector localPoint(XYZPoint point) const;

  /// get the corners 
  inline const std::vector<CLHEP::Hep2Vector> & getCorners() const {return corners_;}

  /// Rescale the Quad to allow for some inaccuracy ...
  void resetCorners();

  /// print
  void print() const;

  /// access methods to the survivalProbability
  inline double survivalProbability() const { return survivalProbability_;};
  inline void setSurvivalProbability(double val) {survivalProbability_=val;};

  /// access to the corners in direction iside; n=0,1
  CLHEP::Hep2Vector& edge(unsigned iside,int n) ;

  /// access to one corner (NE,NW,SE,SW)
  CLHEP::Hep2Vector & edge(CaloDirection);

  /// access to the number
  inline unsigned getNumber() const{return number_;};

  /// get the coordinates in the original frame
  /*
  inline XYZPoint originalCoordinates(CLHEP::Hep2Vector point) const
    {
      XYZPoint p(point.x(),point.y(),0.);
      return trans_.Inverse() * p;
    }
  */
  
  inline bool operator==(const CrystalPad& quad) const
    {
      //      std::cout << " First " << quad.getCellID() << " Second " << this->getCellID() << std::endl;
      return quad.getNumber()==this->getNumber();
    }

  inline bool operator<(const CrystalPad& quad) const
    {
      return (center_.mag()<quad.center().mag());
    }

  /// xmin xmax, ymin ymax of the quad
  void extrems(double &xmin,double& xmax,double &ymin, double& ymax) const;

  ///get the center
  inline const CLHEP::Hep2Vector& center() const {return center_;}

  /// for graphic debugging
  void getDrawingCoordinates(std::vector<float>&x, std::vector<float>&y) const;   

  
 private:

  static std::vector<CLHEP::Hep2Vector> aVector;

  std::vector<CLHEP::Hep2Vector> corners_;
  std::vector<CLHEP::Hep2Vector> dir_;
  unsigned number_; 
  Transform3D trans_;
  ROOT::Math::Rotation3D rotation_;
  XYZVector translation_;
  double survivalProbability_;
  CLHEP::Hep2Vector center_;
  double epsilon_;
  bool dummy_;
  double yscalefactor_;

 public:
  /// equality operator 
    class padEqual
      {
      public:
	padEqual(unsigned cell):ref_(cell) 
	  {
	    //	    std::cout << " quadEqual " << ref_ << std::endl;
	  };
	~padEqual(){;};
	inline bool operator() (const CrystalPad & quad) const
	  {
	    return (ref_==quad.getNumber());
	  }
      private:
	unsigned ref_;
      };
  


};

#include<iosfwd>
std::ostream& operator <<(std::ostream& o ,  CrystalPad & quad);

#endif
