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

#include <cassert>
#include <cmath>
#include <algorithm>

#include "Geometry/CaloGeometry/interface/EcalTrapezoidParameters.h"

typedef EcalTrapezoidParameters::TPFloat    TPFloat    ;
typedef EcalTrapezoidParameters::VertexList VertexList ;
typedef CaloCellGeometry::Pt3D              Pt3D       ;

EcalTrapezoidParameters::EcalTrapezoidParameters(
   TPFloat aHalfLengthXNegZLoY , // bl1, A/2
   TPFloat aHalfLengthXPosZLoY , // bl2
   TPFloat aHalfLengthXPosZHiY , // tl2
   TPFloat aHalfLengthYNegZ    , // h1
   TPFloat aHalfLengthYPosZ    , // h2
   TPFloat aHalfLengthZ        , // dz,  L/2
   TPFloat aAngleAD            , // alfa1
   TPFloat aCoord15X           , // x15
   TPFloat aCoord15Y             // y15
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
   const TPFloat sina1 ( sin( m_a1 ) ) ;
   const TPFloat cosa1 ( cos( m_a1 ) ) ;
   const TPFloat tana1 ( tan( m_a1 - M_PI_2 ) ) ;

   const TPFloat tana4 ( ( m_tl2 - m_bl2 - m_h2*tana1 )/m_h2 ) ;

   m_a4  = M_PI_2 + atan( tana4 ) ;

   m_tl1 = m_bl1 + m_h1*( tana1 + tana4 ) ;

   m_d   = m_h/sina1 ;
   m_D   = m_H/sina1 ;

   const TPFloat tanalp1 ( ( m_D*cosa1 + m_tl1 - m_bl1 )/m_H ) ;
   const TPFloat tanalp2 ( ( m_d*cosa1 + m_tl2 - m_bl2 )/m_h ) ;
   m_alp1 = atan( tanalp1 ) ;
   m_alp2 = atan( tanalp2 ) ;

   const TPFloat sina4 ( sin( m_a4 ) ) ;
   m_c   = m_h/sina4 ;
   m_C   = m_H/sina4 ;
   m_B   = 2*m_tl1 ; // same as m_A - m_D*cosa1 - m_C*cos( m_a4 ) ;

   m_hDd = fabs( m_x15 )*sina1 - m_hAa*cosa1 ;

   const TPFloat xd5 ( ( m_hAa + m_hDd*cosa1 )/sina1 ) ;
   const TPFloat xd6 ( m_D - m_d - xd5 ) ;
   const TPFloat z6  ( sqrt( m_hDd*m_hDd + xd6*xd6 ) ) ;
   TPFloat gb6 ;
   if( 0. == z6 || 1. < fabs( m_hDd/z6 ) )
   {
      gb6 = 0 ;
   }
   else
   {
      gb6 = M_PI - m_a1 - asin( m_hDd/z6 ) ;
   }
   m_hBb = z6*sin( gb6 ) ;

   const TPFloat xb6 ( z6*cos( gb6 ) ) ;
   const TPFloat xb7 ( m_B - xb6 - m_b ) ;
   const TPFloat z7  ( sqrt( m_hBb*m_hBb + xb7*xb7 ) ) ;
   TPFloat gc7 ;
   if( 0 == z7 || 1. < fabs( m_hBb/z7 ) )
   {
      gc7 = 0 ;
   }
   else
   {
      gc7 = M_PI - m_a4 - asin( m_hBb/z7 ) ;   
   }
   m_hCc = z7*sin( gc7 ) ;

   const Pt3D fc ( m_bl2 + m_h2*tanalp2, m_h2, 0 ) ;
   const Pt3D v5 ( m_x15 , m_y15 , -m_L ) ;
   const Pt3D bc ( v5 + Pt3D ( m_bl1 + m_h1*tanalp1, m_h1, 0 ) ) ;
   const Pt3D dc ( fc - bc ) ;

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
TPFloat EcalTrapezoidParameters::dz()    const { return m_dz   ; }
TPFloat EcalTrapezoidParameters::theta() const { return m_th   ; }
TPFloat EcalTrapezoidParameters::phi()   const { return m_ph   ; }
TPFloat EcalTrapezoidParameters::h1()    const { return m_h1   ; }
TPFloat EcalTrapezoidParameters::bl1()   const { return m_bl1  ; }
TPFloat EcalTrapezoidParameters::tl1()   const { return m_tl1  ; }
TPFloat EcalTrapezoidParameters::alp1()  const { return m_alp1 ; }
TPFloat EcalTrapezoidParameters::h2()    const { return m_h2   ; }
TPFloat EcalTrapezoidParameters::bl2()   const { return m_bl2  ; }
TPFloat EcalTrapezoidParameters::tl2()   const { return m_tl2  ; }
TPFloat EcalTrapezoidParameters::alp2()  const { return m_alp2 ; }

TPFloat EcalTrapezoidParameters::x15()   const { return m_x15  ; }
TPFloat EcalTrapezoidParameters::y15()   const { return m_y15  ; }
TPFloat EcalTrapezoidParameters::hAa()   const { return m_hAa  ; }
TPFloat EcalTrapezoidParameters::hBb()   const { return m_hBb  ; }
TPFloat EcalTrapezoidParameters::hCc()   const { return m_hCc  ; }
TPFloat EcalTrapezoidParameters::hDd()   const { return m_hDd  ; }
TPFloat EcalTrapezoidParameters::a1()    const { return m_a1   ; }
TPFloat EcalTrapezoidParameters::a4()    const { return m_a4   ; }
TPFloat EcalTrapezoidParameters::L()     const { return m_L    ; }
TPFloat EcalTrapezoidParameters::a()     const { return m_a    ; }
TPFloat EcalTrapezoidParameters::b()     const { return m_b    ; }
TPFloat EcalTrapezoidParameters::c()     const { return m_c    ; }
TPFloat EcalTrapezoidParameters::d()     const { return m_d    ; }
TPFloat EcalTrapezoidParameters::h()     const { return m_h    ; }
TPFloat EcalTrapezoidParameters::A()     const { return m_A    ; }
TPFloat EcalTrapezoidParameters::B()     const { return m_B    ; }
TPFloat EcalTrapezoidParameters::C()     const { return m_C    ; }
TPFloat EcalTrapezoidParameters::D()     const { return m_D    ; }
TPFloat EcalTrapezoidParameters::H()     const { return m_H    ; }

EcalTrapezoidParameters::VertexList
EcalTrapezoidParameters::vertexList() const
{
   VertexList vtx ;
   vtx.reserve( 8 ) ;

   const TPFloat dztanth ( dz()*tan( theta() ) ) ;

   const TPFloat ph ( phi() ) ;
   const Pt3D  fc ( dztanth*cos(ph), dztanth*sin(ph), dz() ) ;

   const TPFloat h_ ( h() ) ;
   const TPFloat H_ ( H() ) ;
   const TPFloat b_ ( b() ) ;
   const TPFloat B_ ( B() ) ;
   const TPFloat a_ ( a() ) ;
   const TPFloat A_ ( A() ) ;

//   const TPFloat tl1 ( tl1() ) ;

   const TPFloat tanalp1 ( tan(alp1()) ) ;

   const TPFloat tanalp2 ( tan(alp2()) ) ;

   const TPFloat tana1   ( tan( a1() - M_PI_2 )  ) ;

   const Pt3D f1 ( -Pt3D( bl2() + h2()*tanalp2,  h2(), 0 ) ) ;

   const Pt3D f2 ( Pt3D( -h_*tana1, h_, 0 ) + f1 ) ;

   const Pt3D f3 ( f2 + Pt3D( b_,0,0 ) ) ;

   const Pt3D f4 ( Pt3D( a_,0,0 ) + f1 ) ;


   const Pt3D f5 ( -Pt3D( bl1() + h1()*tanalp1,  h1(),      0 ) ) ;

   const Pt3D f6 ( Pt3D( -H_*tana1, H_, 0 ) + f5 ) ;

   const Pt3D f7 ( f6 + Pt3D( B_,0,0 ) ) ;

   const Pt3D f8 ( Pt3D( A_,0,0 ) + f5 ) ;

   vtx.emplace_back(  fc + f1 ) ;
   vtx.emplace_back(  fc + f2 ) ;
   vtx.emplace_back(  fc + f3 ) ;
   vtx.emplace_back(  fc + f4 ) ;
   vtx.emplace_back( -fc + f5 ) ;
   vtx.emplace_back( -fc + f6 ) ;
   vtx.emplace_back( -fc + f7 ) ;
   vtx.emplace_back( -fc + f8 ) ;

   return vtx ;
}
//
// static member functions
//
