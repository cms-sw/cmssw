// @(#)root/mathcore:$Name: V02-02-29 $:$Id: Transform3DPJ.cc,v 1.2 2008/01/22 20:41:27 muzaffar Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// implementation file for class Transform3D
//
// Created by: Lorenzo Moneta  October 27 2005
//
//

#include "FastSimulation/CaloGeometryTools/interface/Transform3DPJ.h"
#include "Math/GenVector/Plane3D.h"

#include <cmath>
#include <algorithm>




namespace ROOT {

namespace Math {


typedef Transform3DPJ::Point  XYZPoint; 
typedef Transform3DPJ::Vector XYZVector; 


// ========== Constructors and Assignment =====================



// construct from two ref frames
Transform3DPJ::Transform3DPJ(const XYZPoint & fr0, const XYZPoint & fr1, const XYZPoint & fr2,
                         const XYZPoint & to0, const XYZPoint & to1, const XYZPoint & to2 )
{
   // takes impl. from CLHEP ( E.Chernyaev). To be checked
   
   XYZVector x1,y1,z1, x2,y2,z2;
   x1 = (fr1 - fr0).Unit();
   y1 = (fr2 - fr0).Unit();
   x2 = (to1 - to0).Unit();
   y2 = (to2 - to0).Unit();
   
   //   C H E C K   A N G L E S
   
   double cos1, cos2;
   cos1 = x1.Dot(y1);
   cos2 = x2.Dot(y2);
   
   if (std::fabs(1.0-cos1) <= 0.000001 || std::fabs(1.0-cos2) <= 0.000001) {
      std::cerr << "Transform3DPJ: Error : zero angle between axes" << std::endl;
      SetIdentity();
   } else {
      if (std::fabs(cos1-cos2) > 0.000001) {
         std::cerr << "Transform3DPJ: Warning: angles between axes are not equal"
         << std::endl;
      }
      
      //   F I N D   R O T A T I O N   M A T R I X
      
      z1 = (x1.Cross(y1)).Unit();
      y1  = z1.Cross(x1);
      
      z2 = (x2.Cross(y2)).Unit();
      y2  = z2.Cross(x2);

      double x1x = x1.X(), x1y = x1.Y(), x1z = x1.Z();
      double y1x = y1.X(), y1y = y1.Y(), y1z = y1.Z();
      double z1x = z1.X(), z1y = z1.Y(), z1z = z1.Z();
      
      double detxx =  (y1y*z1z - z1y*y1z);
      double detxy = -(y1x*z1z - z1x*y1z);
      double detxz =  (y1x*z1y - z1x*y1y);
      double detyx = -(x1y*z1z - z1y*x1z);
      double detyy =  (x1x*z1z - z1x*x1z);
      double detyz = -(x1x*z1y - z1x*x1y);
      double detzx =  (x1y*y1z - y1y*x1z);
      double detzy = -(x1x*y1z - y1x*x1z);
      double detzz =  (x1x*y1y - y1x*x1y);

      double x2x = x2.X(), x2y = x2.Y(), x2z = x2.Z();
      double y2x = y2.X(), y2y = y2.Y(), y2z = y2.Z();
      double z2x = z2.X(), z2y = z2.Y(), z2z = z2.Z();

      double txx = x2x*detxx + y2x*detyx + z2x*detzx;
      double txy = x2x*detxy + y2x*detyy + z2x*detzy;
      double txz = x2x*detxz + y2x*detyz + z2x*detzz;
      double tyx = x2y*detxx + y2y*detyx + z2y*detzx;
      double tyy = x2y*detxy + y2y*detyy + z2y*detzy;
      double tyz = x2y*detxz + y2y*detyz + z2y*detzz;
      double tzx = x2z*detxx + y2z*detyx + z2z*detzx;
      double tzy = x2z*detxy + y2z*detyy + z2z*detzy;
      double tzz = x2z*detxz + y2z*detyz + z2z*detzz;
      
      //   S E T    T R A N S F O R M A T I O N
      
      double dx1 = fr0.X(), dy1 = fr0.Y(), dz1 = fr0.Z();
      double dx2 = to0.X(), dy2 = to0.Y(), dz2 = to0.Z();
      
      SetComponents(txx, txy, txz, dx2-txx*dx1-txy*dy1-txz*dz1,
                    tyx, tyy, tyz, dy2-tyx*dx1-tyy*dy1-tyz*dz1,
                    tzx, tzy, tzz, dz2-tzx*dx1-tzy*dy1-tzz*dz1);
   }
}


// inversion (from CLHEP)
void Transform3DPJ::Invert()
{
   //
   // Name: Transform3DPJ::inverse                     Date:    24.09.96
   // Author: E.Chernyaev (IHEP/Protvino)            Revised:
   //
   // Function: Find inverse affine transformation.
   
   double detxx = fM[kYY]*fM[kZZ] - fM[kYZ]*fM[kZY];
   double detxy = fM[kYX]*fM[kZZ] - fM[kYZ]*fM[kZX];
   double detxz = fM[kYX]*fM[kZY] - fM[kYY]*fM[kZX];
   double det   = fM[kXX]*detxx - fM[kXY]*detxy + fM[kXZ]*detxz;
   if (det == 0) {
      std::cerr << "Transform3DPJ::inverse error: zero determinant" << std::endl;
      return;
   }
   det = 1./det; detxx *= det; detxy *= det; detxz *= det;
   double detyx = (fM[kXY]*fM[kZZ] - fM[kXZ]*fM[kZY] )*det;
   double detyy = (fM[kXX]*fM[kZZ] - fM[kXZ]*fM[kZX] )*det;
   double detyz = (fM[kXX]*fM[kZY] - fM[kXY]*fM[kZX] )*det;
   double detzx = (fM[kXY]*fM[kYZ] - fM[kXZ]*fM[kYY] )*det;
   double detzy = (fM[kXX]*fM[kYZ] - fM[kXZ]*fM[kYX] )*det;
   double detzz = (fM[kXX]*fM[kYY] - fM[kXY]*fM[kYX] )*det;
   SetComponents
      (detxx, -detyx,  detzx, -detxx*fM[kDX]+detyx*fM[kDY]-detzx*fM[kDZ],
       -detxy,  detyy, -detzy,  detxy*fM[kDX]-detyy*fM[kDY]+detzy*fM[kDZ],
       detxz, -detyz,  detzz, -detxz*fM[kDX]+detyz*fM[kDY]-detzz*fM[kDZ]);
}


// get rotations and translations
void Transform3DPJ::GetDecomposition ( Rotation3D &r, XYZVector &v) const
{
   // decompose a trasfomation in a 3D rotation and in a 3D vector (cartesian coordinates) 
   r.SetComponents( fM[kXX], fM[kXY], fM[kXZ],
                    fM[kYX], fM[kYY], fM[kYZ],
                    fM[kZX], fM[kZY], fM[kZZ] );
   
   v.SetCoordinates( fM[kDX], fM[kDY], fM[kDZ] );
}

// transformation on Position Vector (rotation + translations)
XYZPoint Transform3DPJ::operator() (const XYZPoint & p) const
{
   // pass through rotation class (could be implemented directly to be faster)
   
   Rotation3D r;
   XYZVector  t;
   GetDecomposition(r, t);
   XYZPoint pnew = r(p);
   pnew += t;
   return pnew;
}

// transformation on Displacement Vector (only rotation)
XYZVector Transform3DPJ::operator() (const XYZVector & v) const
{
   // pass through rotation class ( could be implemented directly to be faster)
   
   Rotation3D r;
   XYZVector  t;
   GetDecomposition(r, t);
   // only rotation
   return r(v);
}

Transform3DPJ & Transform3DPJ::operator *= (const Transform3DPJ  & t)
{
   // combination of transformations
   
   SetComponents(fM[kXX]*t.fM[kXX]+fM[kXY]*t.fM[kYX]+fM[kXZ]*t.fM[kZX],
                 fM[kXX]*t.fM[kXY]+fM[kXY]*t.fM[kYY]+fM[kXZ]*t.fM[kZY],
                 fM[kXX]*t.fM[kXZ]+fM[kXY]*t.fM[kYZ]+fM[kXZ]*t.fM[kZZ],
                 fM[kXX]*t.fM[kDX]+fM[kXY]*t.fM[kDY]+fM[kXZ]*t.fM[kDZ]+fM[kDX],
                 
                 fM[kYX]*t.fM[kXX]+fM[kYY]*t.fM[kYX]+fM[kYZ]*t.fM[kZX],
                 fM[kYX]*t.fM[kXY]+fM[kYY]*t.fM[kYY]+fM[kYZ]*t.fM[kZY],
                 fM[kYX]*t.fM[kXZ]+fM[kYY]*t.fM[kYZ]+fM[kYZ]*t.fM[kZZ],
                 fM[kYX]*t.fM[kDX]+fM[kYY]*t.fM[kDY]+fM[kYZ]*t.fM[kDZ]+fM[kDY],
                 
                 fM[kZX]*t.fM[kXX]+fM[kZY]*t.fM[kYX]+fM[kZZ]*t.fM[kZX],
                 fM[kZX]*t.fM[kXY]+fM[kZY]*t.fM[kYY]+fM[kZZ]*t.fM[kZY],
                 fM[kZX]*t.fM[kXZ]+fM[kZY]*t.fM[kYZ]+fM[kZZ]*t.fM[kZZ],
                 fM[kZX]*t.fM[kDX]+fM[kZY]*t.fM[kDY]+fM[kZZ]*t.fM[kDZ]+fM[kDZ]);
   
   return *this;
}

void Transform3DPJ::SetIdentity()
{
   //set identity ( identity rotation and zero translation)
   fM[kXX] = 1.0;  fM[kXY] = 0.0; fM[kXZ] = 0.0; fM[kDX] = 0.0;
   fM[kYX] = 0.0;  fM[kYY] = 1.0; fM[kYZ] = 0.0; fM[kDY] = 0.0;
   fM[kZX] = 0.0;  fM[kZY] = 0.0; fM[kZZ] = 1.0; fM[kDZ] = 0.0;
}


void Transform3DPJ::AssignFrom (const Rotation3D  & r,  const XYZVector & v)
{
   // assignment  from rotation + translation
   
   double rotData[9];
   r.GetComponents(rotData, rotData +9);
   // first raw
   for (int i = 0; i < 3; ++i)
      fM[i] = rotData[i];
   // second raw
   for (int i = 0; i < 3; ++i)
      fM[kYX+i] = rotData[3+i];
   // third raw
   for (int i = 0; i < 3; ++i)
      fM[kZX+i] = rotData[6+i];
   
   // translation data
   double vecData[3];
   v.GetCoordinates(vecData, vecData+3);
   fM[kDX] = vecData[0];
   fM[kDY] = vecData[1];
   fM[kDZ] = vecData[2];
}


void Transform3DPJ::AssignFrom(const Rotation3D & r)
{
   // assign from only a rotation  (null translation)
   double rotData[9];
   r.GetComponents(rotData, rotData +9);
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j)
         fM[4*i + j] = rotData[3*i+j];
      // empty vector data
      fM[4*i + 3] = 0;
   }
}

void Transform3DPJ::AssignFrom(const XYZVector & v)
{
   // assign from a translation only (identity rotations)
   fM[kXX] = 1.0;  fM[kXY] = 0.0; fM[kXZ] = 0.0; fM[kDX] = v.X();
   fM[kYX] = 0.0;  fM[kYY] = 1.0; fM[kYZ] = 0.0; fM[kDY] = v.Y();
   fM[kZX] = 0.0;  fM[kZY] = 0.0; fM[kZZ] = 1.0; fM[kDZ] = v.Z();
}

Plane3D Transform3DPJ::operator() (const Plane3D & plane) const
{
   // transformations on a 3D plane
   XYZVector n = plane.Normal();
   // take a point on the plane. Use origin projection on the plane
   // ( -ad, -bd, -cd) if (a**2 + b**2 + c**2 ) = 1
   double d = plane.HesseDistance();
   XYZPoint p( - d * n.X() , - d *n.Y(), -d *n.Z() );
   return Plane3D ( operator() (n), operator() (p) );
}

std::ostream & operator<< (std::ostream & os, const Transform3DPJ & t)
{
   // TODO - this will need changing for machine-readable issues
   //        and even the human readable form needs formatiing improvements
   
   double m[12];
   t.GetComponents(m, m+12);
   os << "\n" << m[0] << "  " << m[1] << "  " << m[2] << "  " << m[3] ;
   os << "\n" << m[4] << "  " << m[5] << "  " << m[6] << "  " << m[7] ;
   os << "\n" << m[8] << "  " << m[9] << "  " << m[10]<< "  " << m[11] << "\n";
   return os;
}

}  // end namespace Math
}  // end namespace ROOT
