#ifndef FastSimulation_CaloGeometryTools_CrystalPad
#define FastSimulation_CaloGeometryTools_CrystalPad

#include "Geometry/CaloTopology/interface/CaloDirection.h"

//CLHEP headers
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include "CLHEP/Vector/TwoVector.h"

//C++ headers
#include <vector>


class CrystalPad
{
 public:
  CrystalPad() { dummy_ = true;};
  /// Order matters. 1234 2341 3412 4123 are ok but not 1324 ....
  CrystalPad(unsigned number, const std::vector<Hep2Vector>& corners);
  /// Constructor from space points, with the description of the local frame (origin,vec1,vec2) where vec1 is normal to the plane and vec2 in the plane

  CrystalPad(unsigned number, int onEcal, const std::vector<HepPoint3D>& corners,HepPoint3D origin, HepVector3D vec1,HepVector3D vec2);
  CrystalPad(unsigned number, const std::vector<HepPoint3D>& corners,const HepTransform3D&,double scaf=1.);
  ~CrystalPad(){;};
  
  /// Check that the point (in the local frame) is inside the crystal. 
  bool inside(const Hep2Vector & point,bool debug=false) const;
  /// Check that the point (in the global frame) is inside the crystal. 
  bool globalinside(HepPoint3D) const;

  /// coordinates of the point in the local frame
  Hep2Vector localPoint(HepPoint3D point) const;

  /// get the corners 
  inline const std::vector<Hep2Vector> & getCorners() const {return corners_;}

  /// Rescale the Quad to allow for some inaccuracy ...
  void resetCorners();

  /// print
  void print() const;

  /// access methods to the survivalProbability
  inline double survivalProbability() const { return survivalProbability_;};
  inline void setSurvivalProbability(double val) {survivalProbability_=val;};

  /// access to the corners in direction iside; n=0,1
  Hep2Vector& edge(unsigned iside,int n) ;

  /// access to one corner (NE,NW,SE,SW)
  Hep2Vector & edge(CaloDirection);

  /// access to the number
  inline unsigned getNumber() const{return number_;};

  /// get the coordinates in the original frame
  inline HepPoint3D originalCoordinates(Hep2Vector point) const
    {
      HepPoint3D p(point);
      return p.transform(trans_.inverse());
    }
  
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
  inline const Hep2Vector& center() const {return center_;}

  
 private:
  std::vector<Hep2Vector> corners_;
  std::vector<Hep2Vector> dir_;
  unsigned number_; 
  HepTransform3D trans_;
  double survivalProbability_;
  Hep2Vector center_;
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
