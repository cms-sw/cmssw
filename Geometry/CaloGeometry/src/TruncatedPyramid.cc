#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
//#include <algorithm>
#include <iostream>
//#include "assert.h"

//----------------------------------------------------------------------

TruncatedPyramid::TruncatedPyramid()
 {}

//----------------------------------------------------------------------

TruncatedPyramid::TruncatedPyramid(double dz, double theta, double phi, 
                                   double h1, double bl1, double tl1, double alpha1, 
                                   double h2, double bl2, double tl2, double alpha2)
{
  bool frontSideIsPositiveZ = (trapeziumArea(h2,tl2,bl2) < trapeziumArea(h1,tl1,bl1));
  init(dz,theta,phi,h1,bl1,tl1,alpha1,h2,bl2,tl2,alpha2,frontSideIsPositiveZ);
  // calculate the center of the front face
  int i;
  int offset;
  if (frontSideIsPositiveZ)
    offset = 4;                 // calculate the center of the front
                                // face from the second four points
  else
    offset = 0;                 // or from the first four points

  HepGeom::Point3D<double> position;
  for (i=0; i<4; ++i)
    position += HepGeom::Point3D<double>(corners[i + offset].x(),corners[i + offset].y(),corners[i + offset].z());

  //While waiting for *= operator
  position *= 0.25;
  setPosition(GlobalPoint(position.x(),position.y(),position.z()));
}

//----------------------------------------------------------------------

void TruncatedPyramid::init(double dz, double theta, double phi, 
                                   double h1, double bl1, double tl1, double alpha1, 
                                   double h2, double bl2, double tl2, double alpha2, 
                                   bool frontSideIsPositiveZ)
{
  corners.resize(8);
  
  double tan_alpha1 = tan(alpha1); // lower plane
  double tan_alpha2 = tan(alpha2); // upper plane

  double tan_theta_cos_phi = tan(theta) * cos(phi);
  double tan_theta_sin_phi = tan(theta) * sin(phi);

  //                       shift due to trapezoid| shift due to the                                               // approximate coordinate 
  //                       axis not parallel to  | fact that the top                                              // signs
  //                       z axis                | and bottom lines
  //                                             | of the trapezium
  //                                             | don't have their
  //                                             | center at the same 
  //                                             | x value (alpha !=0)                                            
  corners[0] = GlobalPoint(-dz*tan_theta_cos_phi - h1 * tan_alpha1       - bl1, -dz*tan_theta_sin_phi - h1 , -dz); // (-,-,-)
  corners[1] = GlobalPoint(-dz*tan_theta_cos_phi + h1 * tan_alpha1       - tl1, -dz*tan_theta_sin_phi + h1 , -dz); // (-,+,-)
  corners[2] = GlobalPoint(-dz*tan_theta_cos_phi + h1 * tan_alpha1       + tl1, -dz*tan_theta_sin_phi + h1 , -dz); // (+,+,-)
  corners[3] = GlobalPoint(-dz*tan_theta_cos_phi - h1 * tan_alpha1       + bl1, -dz*tan_theta_sin_phi - h1 , -dz); // (+,-,-)
                                                                             
  corners[4] = GlobalPoint(dz*tan_theta_cos_phi  - h2 * tan_alpha2       - bl2,  dz*tan_theta_sin_phi - h2  , dz); // (-,-,+)
  corners[5] = GlobalPoint(dz*tan_theta_cos_phi  + h2 * tan_alpha2       - tl2,  dz*tan_theta_sin_phi + h2  , dz); // (-,+,+)
  corners[6] = GlobalPoint(dz*tan_theta_cos_phi  + h2 * tan_alpha2       + tl2,  dz*tan_theta_sin_phi + h2  , dz); // (+,+,+)
  corners[7] = GlobalPoint(dz*tan_theta_cos_phi  - h2 * tan_alpha2       + bl2,  dz*tan_theta_sin_phi - h2  , dz); // (+,-,+)

  // corners[0],corners[3] and corners[1],corners[2] make the parallel lines of the -dz trapezium
  // corners[4],corners[7] and corners[5],corners[6] make the parallel lines of the +dz trapezium

  // determine which one is the front face

  
  //--------------------

  thetaAxis = theta;
  phiAxis = phi;

}

//----------------------------------------------------------------------

// bool 
// TruncatedPyramid::inside(const HepGeom::Point3D<double> &testPoint) const
// {
//   vector<HepPlane3D> boundaries=getBoundaries();

//   const HepGeom::Point3D<float> testPointb(testPoint.x(),testPoint.y(),testPoint.z());
//   // with ordered planes this method becomes simpler
//   if ((testPoint-boundaries[0].point(testPointb))*(testPoint-boundaries[1].point(testPointb))>0){
//    return false ;
//   } 
//   if ((testPoint-boundaries[2].point(testPointb))*(testPoint-boundaries[5].point(testPointb))>0){   
//    return false ;
//   } 
//   if ((testPoint-boundaries[3].point(testPointb))*(testPoint-boundaries[4].point(testPointb))>0){ 
//    return false ;
//   }
//   return true;
// }

// //For the moment returning them only when asked. If speed will be a concern move them as protected members
// const vector<HepPlane3D> & TruncatedPyramid::getBoundaries() const
// { 
//   vector<HepPlane3D> boundaries;
//   // create the boundary planes, the first boundary plane will be the
//   // front side
//   if (frontSideIsPositiveZ)
//     {
//       boundaries.push_back(HepPlane3D(corners[4], corners[5], corners[6])) ;  // trapezium at +dz
//       boundaries.push_back(HepPlane3D(corners[0], corners[1], corners[2])) ; // trapezium at -dz
//     }
//   else
//     {
//       boundaries.push_back(HepPlane3D(corners[0], corners[1], corners[2])) ; // trapezium at -dz
//       boundaries.push_back(HepPlane3D(corners[4], corners[5], corners[6])) ;  // trapezium at +dz
//     }
  
//   // now generate the other boundaries
//   boundaries.push_back(HepPlane3D(corners[0], corners[1], corners[4])); // (-,-,-) (-,+,-) (-,-,+) -> (-,?,?)
//   boundaries.push_back(HepPlane3D(corners[1], corners[2], corners[5])); // (-,+,-) (+,+,-) (-,+,+) -> (?,+,?)
//   boundaries.push_back(HepPlane3D(corners[0], corners[3], corners[4])); // (-,-,-) (+,-,-) (-,-,+) -> (?,-,?)
//   boundaries.push_back(HepPlane3D(corners[2], corners[3], corners[6])); // (+,+,-) (+,-,-) (+,+,+) -> (+,?,?)  
  
//   // normalize the planes
//   for (unsigned i=0; i < boundaries.size(); ++i)
//     boundaries[i] = boundaries[i].normalize();
   
//   return boundaries ; 
// }

const vector<GlobalPoint> & TruncatedPyramid::getCorners() const
{ return corners ; }

//----------------------------------------------------------------------
// void 
// TruncatedPyramid::hepTransform(const HepTransform3D &transformation)
// {
//   unsigned int i;


//   for (i=0; i<corners.size(); ++i)
//     corners[i].transform(transformation);
  
//   position_.transform(transformation);

//   HepVector3D axe(1.,1.,1.); 
//   axe.setMag(1.); // must do this first
//   axe.setTheta(thetaAxis);
//   axe.setPhi(phiAxis);

//   axe.transform(transformation);
//   thetaAxis = axe.getTheta();
//   phiAxis = axe.getPhi();
// }

//----------------------------------------------------------------------
double 
TruncatedPyramid::trapeziumArea(double halfHeight, double
                                halfTopLength, double
                                halfBottomLength)
{
  return (halfTopLength + halfTopLength) * halfHeight * 2;
}



void TruncatedPyramid::dump(const char * prefix="") const {

  // a crystal must have eight corners (not only the front face...)
  assert(getCorners().size() == 8);
  cout << prefix << "Center: " <<  getPosition() << endl;
  float thetaaxis_= getThetaAxis();
  float phiaxis_= getPhiAxis();
  cout << prefix << "Axis: " <<  thetaaxis_ << " " << phiaxis_ << endl;
  //  vector<HepPoint3D> xtCorners=getCorners();
  for ( unsigned int  ci=0; ci !=corners.size(); ci++) {
    cout << prefix << "Corner: " << corners[ci] << endl;
  }
}
//----------------------------------------------------------------------
