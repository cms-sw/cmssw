#ifndef PreshowerStrip_h
#define PreshowerStrip_h

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Transform3D.h>
#include <vector>


/**

   \class PreshowerStrip

   \brief A base class to handle the shape of preshower strips.

$Date: 2006/10/26 08:57:10 $
$Revision: 1.2 $
\author F. Cossutti
   
*/


class PreshowerStrip : public CaloCellGeometry
{
public:

  PreshowerStrip() ;

  PreshowerStrip(double dx, double dy, double dz);

  virtual ~PreshowerStrip(){};

  //! Inside the volume?
  virtual bool inside(const GlobalPoint & point) const;  
  
  //! Access to data
  virtual const std::vector<GlobalPoint> & getCorners() const;  

  /** Transform (e.g. move or rotate) this box.
      Transforms the corner points and the reference point.
  */
  void hepTransform(const HepTransform3D &transformation);

protected:
  
  //! Keep corners info
  std::vector<GlobalPoint> corners;

 private:

  double dx_;
  double dy_;
  double dz_;

};

std::ostream& operator<<(std::ostream& s,const PreshowerStrip& cell);
#endif
