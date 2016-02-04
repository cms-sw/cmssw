#ifndef VectorFieldInterpolation_H
#define VectorFieldInterpolation_H

/** \class VectorFieldInterpolation
 *
 * linear interpolation of a field (3D) in space (3D)
 *
 * \author : <Volker.Drollinger@cern.ch>
 * $date   : 04/08/2003 14:46:10 CET $
 *
 * Modifications:
 * droll: change from float to double
 * $date   : 08/09/2003 18:35:10 CET $
 * droll: rename methods according to CMS coding rules
 * $date   : 22/09/2003 11:15:38 CET $
 *
 */

// ***************************************************************
// by droll (19/07/03)
//
// basic equation (1D example):  y = y0 + (x-x0)/(x1-x0)*(y1-y0)
// find field value y at point x
// corner points x0 and x1 with field  values
// field values y0 and y1 at points x0 and x1
//
// input should be organized like in the following sketch
// 3D interpolation cell (example Cartesian coordinates):
//
//          (011) *-------------------* (111)
//               /|                  /|
//              / |                 / |
//             /  |                /  |
//            /   |               /   |
//           /    |              /    |
//          /     |             /     |
//   (010) *-------------------* (110)|
//         |      |            |      |
//         |      |            |      |
//         |(001) *------------|------* (101)
//         |     /             |     /
//         |    /              |    /
//         |   /               |   /
//         |  /                |  /
//         | /                 | /
//         |/                  |/
//   (000) *-------------------* (100)
//
// 1. iteration: interpolation cell  -> interpolation plane
//
// 2. iteration: interpolation plane -> interpolation line
//
// 3. iteration: interpolation line  -> interpolation at SC[3]
//
// ***************************************************************

class VectorFieldInterpolation{
public:
  // constructor
  VectorFieldInterpolation(){}
  // destructor
  ~VectorFieldInterpolation(){}

private:
  // spatial coordinates, where the field has to be calculated
  //                X1 ,  X2 , X3
  double SC[3]; // {0.0 ,0.0 ,0.0 };
  
  // values describing the 8 corners of an interpolation cell
  // 6 dimensions: 3 space dimensions + 3 field dimensions
  //                          X1 ,  X2 ,  X3 ,  F1 ,  F2 ,  F3
  double CellPoint000[6]; // {0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 };
  double CellPoint100[6];
  double CellPoint010[6];
  double CellPoint110[6];
  double CellPoint001[6];
  double CellPoint101[6];
  double CellPoint011[6];
  double CellPoint111[6];

  // 3 components of the interpolated vector field at spatial coordinates SC
  //                F1  , F2  , F3
  double VF[3]; // {0.0 , 0.0 , 0.0 };


public:
   // Accessors
   /// provide the interpolation algorithm with 8 points, where the field is known (in)
   void defineCellPoint000(double X1, double X2, double X3, double  F1, double  F2, double  F3);
   void defineCellPoint100(double X1, double X2, double X3, double  F1, double  F2, double  F3);
   void defineCellPoint010(double X1, double X2, double X3, double  F1, double  F2, double  F3);
   void defineCellPoint110(double X1, double X2, double X3, double  F1, double  F2, double  F3);
   void defineCellPoint001(double X1, double X2, double X3, double  F1, double  F2, double  F3);
   void defineCellPoint101(double X1, double X2, double X3, double  F1, double  F2, double  F3);
   void defineCellPoint011(double X1, double X2, double X3, double  F1, double  F2, double  F3);
   void defineCellPoint111(double X1, double X2, double X3, double  F1, double  F2, double  F3);
   /// receive the interpolated field (out) at any point in space (in)
   void putSCoordGetVField(double X1, double X2, double X3, double &F1, double &F2, double &F3);
};

#endif
