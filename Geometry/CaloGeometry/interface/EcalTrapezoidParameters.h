#if !defined(ECALCOMMONDATA_ECALTRAPEZOIDPARAMETERS_H)
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
// system include files

// user include files

#include <vector>
#include <CLHEP/Geometry/Point3D.h>

// forward declarations


class EcalTrapezoidParameters
{
      // ---------- friend classes and functions ---------------
   public:
      // ---------- constants, enums and typedefs --------------

      typedef std::vector<HepGeom::Point3D<double> > VertexList ;

      // ---------- Constructors and destructor ----------------

      EcalTrapezoidParameters( double aHalfLengthXNegZLoY , // bl1, A/2
			       double aHalfLengthXPosZLoY , // bl2, a/2
			       double aHalfLengthXPosZHiY , // tl2, b/2
			       double aHalfLengthYNegZ    , // h1,  H/2
			       double aHalfLengthYPosZ    , // h2,  h/2
			       double aHalfLengthZ        , // dz,  L/2
			       double aAngleAD            , // alfa1
			       double aCoord15X           , // x15
			       double aCoord15Y             // y15
	 ) ;

      //virtual ~EcalTrapezoidParameters() ;

      // ---------- member functions ---------------------------
      // ---------- const member functions ---------------------

      // GEANT parameters, in order
      double dz()    const ;
      double theta() const ;
      double phi()   const ;
      double h1()    const ;
      double bl1()   const ;
      double tl1()   const ;
      double alp1()  const ;
      double h2()    const ;
      double bl2()   const ;
      double tl2()   const ;
      double alp2()  const ;

      // everything else
      double x15()   const ;
      double y15()   const ;
      double hAa()   const ;
      double hBb()   const ;
      double hCc()   const ;
      double hDd()   const ;
      double a1()    const ;
      double a4()    const ;

      double L()     const ;
      double a()     const ;
      double b()     const ;
      double c()     const ;
      double d()     const ;
      double h()     const ;
      double A()     const ;
      double B()     const ;
      double C()     const ;
      double D()     const ;
      double H()     const ;

      VertexList vertexList() const ; // order is as in picture above: index=vtx-1

      // ---------- static member functions --------------------
   protected:
      // ---------- protected member functions -----------------
      // ---------- protected const member functions -----------
   private:
      // ---------- Constructors and destructor ----------------
      EcalTrapezoidParameters();
      EcalTrapezoidParameters( const EcalTrapezoidParameters& ); // stop default

      // ---------- assignment operator(s) ---------------------
      const EcalTrapezoidParameters& operator=( const EcalTrapezoidParameters& ); // stop default

      // ---------- private member functions -------------------
      // ---------- private const member functions -------------
      // ---------- data members -------------------------------

      double m_dz, m_th, m_ph, m_h1, m_bl1, m_tl1, m_alp1, 
	                       m_h2, m_bl2, m_tl2, m_alp2 ;
      double m_a1, m_hAa, m_x15, m_y15 ;
      double m_a4, m_hBb, m_hCc, m_hDd ;
      double m_L, m_a, m_b, m_c, m_d, m_h, m_A, m_B, m_C, m_D, m_H ;

      // ---------- static data members ------------------------
};

// inline function definitions

#endif /* ECALCOMMONDATA_ECALTRAPEZOIDPARAMETERS_H */
