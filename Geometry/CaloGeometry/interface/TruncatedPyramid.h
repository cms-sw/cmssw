#ifndef TruncatedPyramid_h
#define TruncatedPyramid_h

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <CLHEP/Geometry/Point3D.h>
/* #include <CLHEP/Geometry/Plane3D.h> */
/* #include <CLHEP/Geometry/Vector3D.h> */
/* #include <CLHEP/Geometry/Transform3D.h> */
#include <vector>


/**

   \class TruncatedPyramid

   \brief A base class to handle the particular shape of Ecal Xtals. Taken from ORCA Calorimetry Code
   
*/

using namespace std;


class TruncatedPyramid : public CaloCellGeometry
{
public:

  TruncatedPyramid() ;

  /** intializes this truncated pyramid from the trapezium
      parameters. The front face is made the trapezium with the
      smaller area */
  TruncatedPyramid(double dz,                 // axis length 
                   double theta, double phi,  // axis direction
                   double h1, double bl1, double tl1, double alpha1, // trapezium face at z < 0 parameters
                   double h2, double bl2, double tl2, double alpha2 // trapezium face at z > 0 parameters
                   );

  virtual ~TruncatedPyramid(){};

/*   //! Inside the volume? */
/*   bool inside(const HepPoint3D &point) const;   */
  
/*   //! Access to data */
/*   virtual const vector<HepPlane3D> & getBoundaries() const; */
  
  //! Access to data
  virtual const vector<GlobalPoint> & getCorners() const;  

  /** Return thetaAxis polar angle of axis of the cristal */
  const float& getThetaAxis() const { return thetaAxis; }

  /** Return phiAxis azimuthal angle of axis of the cristal */
  const float& getPhiAxis() const { return phiAxis; }


  /** Transform (e.g. move or rotate) this truncated
      pyramid. Transforms the boundaries, the corner points etc.
      
      This could eventually go up to CellGeometry as an abstract
      function.
  */
  //  void hepTransform(const HepTransform3D &transformation);

  /** helper function to calculate a trapezium area. This can e.g.
      be used to decide which of the two trapezium faces is the front
      face (usually the smaller one) 

      \param halfHeight half height (distance between the parallel sides)
      \param halfTopLength half side of the 'upper' parallel side
      \param halfBottomLength half side of the 'lower' parallel side
  */
  static double trapeziumArea(double halfHeight, double halfTopLength, double halfBottomLength);

  /// print out the element, with an optional string prefix, maybe OVAL identifier
  virtual void dump(const char * prefix) const;

protected:
  
  //! Keep corners info
  vector<GlobalPoint> corners;
  
  /** Polar angle of the axis of the cristal */
  float thetaAxis;

  /** Azimuthal angle of the axis of teh cristal */
  float phiAxis;

  /** constructs the crystal from the Geant3 like parameters.
      The trapezoid's center is at (0,0,0). 

      For a description of the parameters, see the 
      <A HREF="http://wwwinfo.cern.ch/asdoc/geant_html3/node109.html#SECTION041000000000000000000000">Geant 3 manual section GEOM050</A>, 
      description of the shape 'TRAP'.

      \param frontSideIsPositiveZ The trapezium shaped faces are
        parallel to the x-y plane. This parameter specified whether 
        the side at positive or negative z should be considered as the
        front face of the crystal.

       
  */
  virtual void init(double dz,                 // axis length 
            double theta, double phi,  // axis direction
            double h1, double bl1, double tl1, double alpha1, // trapezium face at z < 0 parameters
            double h2, double bl2, double tl2, double alpha2, // trapezium face at z > 0 parameters
            bool frontSideIsPositiveZ);

};

/* //! utility */
/* HepPoint3D findCrossPoint(const HepPlane3D & pl1, const HepPlane3D & pl2, const HepPlane3D & pl3);    */
  
#endif
