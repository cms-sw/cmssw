#ifndef ECALCOMMONDATA_ECALTRAPEZOIDPARAMETERS_H
#define ECALCOMMONDATA_ECALTRAPEZOIDPARAMETERS_H
// -*- C++ -*-
//
// Package:     EcalCommonData
// Module:      EcalTrapezoidParameters
// 
// Description: Dimensionally, all you need to know about a trapezoid
//
// Usage:
//        This class exists because there are many lengths and
//        angles of interest in a trapezoidal polyhedron,
//        and it makes more sense to put them in one place.
//
//        
//        Our naming convention is as follows: lower case abcdh at +Z
//                                             upper case ABCDH at -Z
//                                             L = full length in Z
//                                             a1, a4 = angles at vertices 1,4
//                                             hXx = perp distance between X, x
//                                             (x15,y15) = coord of v5 wrt v1
//                                 B
//    6____________________________________________________________________ 7
//     \                   /\                                             /
//      \                  |                                           /
//       \                hBb      b                                  /
//        \        2 _____\/__________________________ 3            /
//         \         \              /\               /            /
//          \         \             |              /            /
//           \         \            |            /            /
//            \         \           |          /            /
//          D  \       d \          h        / c          /   C    Y
//              \         \         |      /            /         ^
//               \         \ a1     |    /            /           |
//                \         \______\/__/            /             |
//                 \       1/\     a   4          /               |
//                  \       |                   /                 |_____ X
//                   \      |                 /                   /
//                    \  y15=hAa            /                   /
//                     \    |      a4     /                   /
//                      \__\/___________/                    Z (out of page)
//                    5 |   |   A       8
//                  --->|x15|<----      
//
//
//        Specifying the minimal parameters for a GEANT TRAP
//        requires 9 numbers, *NOT* 11 as used in GEANT or DDD.
//
//        We choose the following 9:
//                    L/2, a/2, b/2, h/2, a1, A/2, H/2, hAa, x15
//        In GEANT these correspond to
//                    dz,  bl2, tl2,  h2,  -, bl1,  h1,   -,   -
//
// Author:      Brian K. Heltsley
// Created:     Wed Aug 12 09:25:08 EDT 1998

#include <vector>
#include <CLHEP/Geometry/Point3D.h>

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

class EcalTrapezoidParameters
{
 public:
 
  using VertexList = CaloCellGeometry::Pt3DVec;
  using TPFloat = CaloCellGeometry::CCGFloat;

  EcalTrapezoidParameters( TPFloat aHalfLengthXNegZLoY , // bl1, A/2
			   TPFloat aHalfLengthXPosZLoY , // bl2, a/2
			   TPFloat aHalfLengthXPosZHiY , // tl2, b/2
			   TPFloat aHalfLengthYNegZ    , // h1,  H/2
			   TPFloat aHalfLengthYPosZ    , // h2,  h/2
			   TPFloat aHalfLengthZ        , // dz,  L/2
			   TPFloat aAngleAD            , // alfa1
			   TPFloat aCoord15X           , // x15
			   TPFloat aCoord15Y             // y15
			   );

  // GEANT parameters, in order
  TPFloat dz()    const ;
  TPFloat theta() const ;
  TPFloat phi()   const ;
  TPFloat h1()    const ;
  TPFloat bl1()   const ;
  TPFloat tl1()   const ;
  TPFloat alp1()  const ;
  TPFloat h2()    const ;
  TPFloat bl2()   const ;
  TPFloat tl2()   const ;
  TPFloat alp2()  const ;
  
  // everything else
  TPFloat x15()   const ;
  TPFloat y15()   const ;
  TPFloat hAa()   const ;
  TPFloat hBb()   const ;
  TPFloat hCc()   const ;
  TPFloat hDd()   const ;
  TPFloat a1()    const ;
  TPFloat a4()    const ;
  
  TPFloat L()     const ;
  TPFloat a()     const ;
  TPFloat b()     const ;
  TPFloat c()     const ;
  TPFloat d()     const ;
  TPFloat h()     const ;
  TPFloat A()     const ;
  TPFloat B()     const ;
  TPFloat C()     const ;
  TPFloat D()     const ;
  TPFloat H()     const ;
  
  VertexList vertexList() const ; // order is as in picture above: index=vtx-1
  
  EcalTrapezoidParameters() = delete;
  EcalTrapezoidParameters( const EcalTrapezoidParameters& ) = delete;
  const EcalTrapezoidParameters& operator=( const EcalTrapezoidParameters& ) = delete;

 private:
  
  TPFloat m_dz, m_th, m_ph, m_h1, m_bl1, m_tl1, m_alp1, m_h2, m_bl2, m_tl2, m_alp2 ;
  TPFloat m_a1, m_hAa, m_x15, m_y15 ;
  TPFloat m_a4, m_hBb, m_hCc, m_hDd ;
  TPFloat m_L, m_a, m_b, m_c, m_d, m_h, m_A, m_B, m_C, m_D, m_H ;
};

#endif /* ECALCOMMONDATA_ECALTRAPEZOIDPARAMETERS_H */
