// -*- C++ -*-
//
// Package:     EcalCommonData
// Module:      EcalTrapezoidParameters
// 
// Description: Do trigonometry to figure out all useful dimensions/angles
//
// Implementation: Compute everything in constructor, put in member data
//
// Author:      Brian K. Heltsley
// Created:     Wed Aug 12 09:24:56 EDT 1998
//

// system include files
#include <assert.h>
#include <cmath>
#include <algorithm>

// user include files
#include "Geometry/CaloGeometry/interface/EcalTrapezoidParameters.h"

// STL classes

//
// constants, enums and typedefs
//

//static const char* const kReport = "DetectorGeometry.DGTrapezoidParameters" ;

//
// static data member definitions
//

//
// constructors and destructor
//
EcalTrapezoidParameters::EcalTrapezoidParameters(
   double aHalfLengthXNegZLoY , // bl1, A/2
   double aHalfLengthXPosZLoY , // bl2
   double aHalfLengthXPosZHiY , // tl2
   double aHalfLengthYNegZ    , // h1
   double aHalfLengthYPosZ    , // h2
   double aHalfLengthZ        , // dz,  L/2
   double aAngleAD            , // alfa1
   double aCoord15X           , // x15
   double aCoord15Y             // y15
   ) 
{
   m_dz  = aHalfLengthZ        ;
   m_h1  = aHalfLengthYNegZ    ;
   m_bl1 = aHalfLengthXNegZLoY ;
   m_h2  = aHalfLengthYPosZ    ;
   m_bl2 = aHalfLengthXPosZLoY ;
   m_tl2 = aHalfLengthXPosZHiY ;

   m_a1  = aAngleAD            ;
   m_y15 = aCoord15Y           ;
   m_x15 = aCoord15X           ;

   m_hAa = fabs( m_y15 ) ;

   m_L   = 2*m_dz  ;
   m_h   = 2*m_h2  ;
   m_a   = 2*m_bl2 ;
   m_b   = 2*m_tl2 ;
   m_H   = 2*m_h1  ;
   m_A   = 2*m_bl1 ;

   // derive everything else
   const double sina1 ( sin( m_a1 ) ) ;
   const double cosa1 ( cos( m_a1 ) ) ;
   const double tana1 ( tan( m_a1 - M_PI_2 ) ) ;

   const double tana4 ( ( m_tl2 - m_bl2 - m_h2*tana1 )/m_h2 ) ;

   m_a4  = M_PI_2 + atan( tana4 ) ;

   m_tl1 = m_bl1 + m_h1*( tana1 + tana4 ) ;

   m_d   = m_h/sina1 ;
   m_D   = m_H/sina1 ;

   const double tanalp1 ( ( m_D*cosa1 + m_tl1 - m_bl1 )/m_H ) ;
   const double tanalp2 ( ( m_d*cosa1 + m_tl2 - m_bl2 )/m_h ) ;
   m_alp1 = atan( tanalp1 ) ;
   m_alp2 = atan( tanalp2 ) ;

   const double sina4 ( sin( m_a4 ) ) ;
   m_c   = m_h/sina4 ;
   m_C   = m_H/sina4 ;
   m_B   = 2*m_tl1 ; // same as m_A - m_D*cosa1 - m_C*cos( m_a4 ) ;

   m_hDd = fabs( m_x15 )*sina1 - m_hAa*cosa1 ;

   const double xd5 ( ( m_hAa + m_hDd*cosa1 )/sina1 ) ;
   const double xd6 ( m_D - m_d - xd5 ) ;
   const double z6  ( sqrt( m_hDd*m_hDd + xd6*xd6 ) ) ;
   double gb6 ;
   if( 0. == z6 || 1. < fabs( m_hDd/z6 ) )
   {
      gb6 = 0 ;
   }
   else
   {
      gb6 = M_PI - m_a1 - asin( m_hDd/z6 ) ;
   }
   m_hBb = z6*sin( gb6 ) ;

   const double xb6 ( z6*cos( gb6 ) ) ;
   const double xb7 ( m_B - xb6 - m_b ) ;
   const double z7  ( sqrt( m_hBb*m_hBb + xb7*xb7 ) ) ;
   double gc7 ;
   if( 0 == z7 || 1. < fabs( m_hBb/z7 ) )
   {
      gc7 = 0 ;
   }
   else
   {
      gc7 = M_PI - m_a4 - asin( m_hBb/z7 ) ;   
   }
   m_hCc = z7*sin( gc7 ) ;

   const HepPoint3D fc ( m_bl2 + m_h2*tanalp2, m_h2, 0 ) ;
   const HepPoint3D v5 ( m_x15 , m_y15 , -m_L ) ;
   const HepPoint3D bc ( v5 +
		   HepPoint3D( m_bl1 + m_h1*tanalp1, m_h1, 0 ) ) ;
   const HepPoint3D dc ( fc - bc ) ;

   m_th  = dc.theta() ;
   m_ph  = dc.phi()   ;
}
//      m_hBb, m_hCc, m_hDd ;

// EcalTrapezoidParameters::EcalTrapezoidParameters( const EcalTrapezoidParameters& rhs )
// {
//    // do actual copying here; if you implemented
//    // operator= correctly, you may be able to use just say      
//    *this = rhs;
// }

//EcalTrapezoidParameters::~EcalTrapezoidParameters()
//{
//}

//
// assignment operators
//
// const EcalTrapezoidParameters& EcalTrapezoidParameters::operator=( const EcalTrapezoidParameters& rhs )
// {
//   if( this != &rhs ) {
//      // do actual copying here, plus:
//      // "SuperClass"::operator=( rhs );
//   }
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//
double EcalTrapezoidParameters::dz()    const { return m_dz   ; }
double EcalTrapezoidParameters::theta() const { return m_th   ; }
double EcalTrapezoidParameters::phi()   const { return m_ph   ; }
double EcalTrapezoidParameters::h1()    const { return m_h1   ; }
double EcalTrapezoidParameters::bl1()   const { return m_bl1  ; }
double EcalTrapezoidParameters::tl1()   const { return m_tl1  ; }
double EcalTrapezoidParameters::alp1()  const { return m_alp1 ; }
double EcalTrapezoidParameters::h2()    const { return m_h2   ; }
double EcalTrapezoidParameters::bl2()   const { return m_bl2  ; }
double EcalTrapezoidParameters::tl2()   const { return m_tl2  ; }
double EcalTrapezoidParameters::alp2()  const { return m_alp2 ; }

double EcalTrapezoidParameters::x15()   const { return m_x15  ; }
double EcalTrapezoidParameters::y15()   const { return m_y15  ; }
double EcalTrapezoidParameters::hAa()   const { return m_hAa  ; }
double EcalTrapezoidParameters::hBb()   const { return m_hBb  ; }
double EcalTrapezoidParameters::hCc()   const { return m_hCc  ; }
double EcalTrapezoidParameters::hDd()   const { return m_hDd  ; }
double EcalTrapezoidParameters::a1()    const { return m_a1   ; }
double EcalTrapezoidParameters::a4()    const { return m_a4   ; }
double EcalTrapezoidParameters::L()     const { return m_L    ; }
double EcalTrapezoidParameters::a()     const { return m_a    ; }
double EcalTrapezoidParameters::b()     const { return m_b    ; }
double EcalTrapezoidParameters::c()     const { return m_c    ; }
double EcalTrapezoidParameters::d()     const { return m_d    ; }
double EcalTrapezoidParameters::h()     const { return m_h    ; }
double EcalTrapezoidParameters::A()     const { return m_A    ; }
double EcalTrapezoidParameters::B()     const { return m_B    ; }
double EcalTrapezoidParameters::C()     const { return m_C    ; }
double EcalTrapezoidParameters::D()     const { return m_D    ; }
double EcalTrapezoidParameters::H()     const { return m_H    ; }

EcalTrapezoidParameters::VertexList
EcalTrapezoidParameters::vertexList() const
{
   VertexList vtx ;
   vtx.reserve( 8 ) ;

   const double dztanth ( dz()*tan( theta() ) ) ;

   const double ph ( phi() ) ;
   const HepPoint3D fc ( dztanth*cos(ph), dztanth*sin(ph), dz() ) ;

   const double h ( h() ) ;
   const double H ( H() ) ;
   const double b ( b() ) ;
   const double B ( B() ) ;
   const double a ( a() ) ;
   const double A ( A() ) ;

//   const double tl1 ( tl1() ) ;

   const double tanalp1 ( tan(alp1()) ) ;

   const double tanalp2 ( tan(alp2()) ) ;

   const double tana1   ( tan( a1() - M_PI_2 )  ) ;

   const HepPoint3D f1 ( -HepPoint3D( bl2() + h2()*tanalp2,  h2(), 0 ) ) ;

   const HepPoint3D f2 ( HepPoint3D( -h*tana1, h, 0 ) + f1 ) ;

   const HepPoint3D f3 ( f2 + HepPoint3D( b,0,0 ) ) ;

   const HepPoint3D f4 ( HepPoint3D( a,0,0 ) + f1 ) ;


   const HepPoint3D f5 ( -HepPoint3D( bl1() + h1()*tanalp1,  h1(),      0 ) ) ;

   const HepPoint3D f6 ( HepPoint3D( -H*tana1, H, 0 ) + f5 ) ;

   const HepPoint3D f7 ( f6 + HepPoint3D( B,0,0 ) ) ;

   const HepPoint3D f8 ( HepPoint3D( A,0,0 ) + f5 ) ;

   vtx.push_back(  fc + f1 ) ;
   vtx.push_back(  fc + f2 ) ;
   vtx.push_back(  fc + f3 ) ;
   vtx.push_back(  fc + f4 ) ;
   vtx.push_back( -fc + f5 ) ;
   vtx.push_back( -fc + f6 ) ;
   vtx.push_back( -fc + f7 ) ;
   vtx.push_back( -fc + f8 ) ;

   return vtx ;
}
//
// static member functions
//
