#include <cmath>
#include "TMath.h"
#include "TEveVector.h"

namespace FWPFMaths
{
//______________________________________________________________________________
float
sgn( float val )
{
   return ( val < 0 ) ? -1 : 1;
}

//______________________________________________________________________________
TEveVector
cross( const TEveVector &v1, const TEveVector &v2 )
{
   TEveVector vec;

   vec.fX = ( v1.fY * v2.fZ ) - ( v1.fZ * v2.fY );
   vec.fY = ( v1.fZ * v2.fX ) - ( v1.fX * v2.fZ );
   vec.fZ = ( v1.fX * v2.fY ) - ( v1.fY * v2.fX );

   return vec;
}

//______________________________________________________________________________
float
dot( const TEveVector &v1, const TEveVector &v2 )
{
   float result = ( v1.fX * v2.fX ) + ( v1.fY * v2.fY ) + ( v1.fZ * v1.fZ );

   return result;
}

//______________________________________________________________________________
TEveVector
lineCircleIntersect( const TEveVector &v1, const TEveVector &v2, float r )
{
   // Definitions
   float x, y;
   float dx = v1.fX - v2.fX;
   float dy = v1.fY - v2.fY;
   float dr = sqrt( ( dx * dx ) + ( dy * dy ) );
   float D = ( ( v2.fX * v1.fY ) - ( v1.fX * v2.fY ) );

   float rtDescrim = sqrt( ( ( r * r ) * ( dr * dr ) ) - ( D * D ) );

   if( dy < 0 )   // Going down
   {
      x = ( D * dy ) - ( ( sgn(dy) * dx ) * rtDescrim );
      x /= ( dr * dr );

      y = ( -D * dx ) - ( fabs( dy ) * rtDescrim );
      y /= ( dr * dr );
   }
   else
   {

      x = ( D * dy ) + ( ( sgn(dy) * dx ) * rtDescrim );
      x /= ( dr * dr );

      y = ( -D * dx ) + ( fabs( dy ) * rtDescrim );
      y /= ( dr * dr );
   }

   TEveVector result = TEveVector( x, y, 0.001 );
   return result;
}

//______________________________________________________________________________
TEveVector
lineLineIntersect( const TEveVector &p1, const TEveVector &p2, const TEveVector &p3, const TEveVector &p4 )
{
   TEveVector a = p2 - p1;
   TEveVector b = p4 - p3;
   TEveVector c = p3 - p1;
   TEveVector result;
   float s, val;

   s = dot( cross( c, b ), cross( a, b ) );  
   val = dot( cross( a, b ), cross( a, b ) );
   s /= val;
   
   result = p1 + ( a * s );
   return result; 
}

//______________________________________________________________________________
float
linearInterpolation( const TEveVector &p1, const TEveVector &p2, float z )
{
   float y;

   y = ( ( z - fabs( p1.fZ ) ) * p2.fY ) + ( ( fabs( p2.fZ ) - z ) * p1.fY );
   y /= ( fabs( p2.fZ) - fabs( p1.fZ ) );

   return y;
}

//______________________________________________________________________________
bool
checkIntersect( const TEveVector &p, float r )
{
   float h = sqrt( ( p.fX * p.fX ) + ( p.fY * p.fY ) );
   
   if( h >= r )
      return true;

   return false;
}

//______________________________________________________________________________
float
calculateEt( const TEveVector &centre, float e )
{
   TEveVector vec = centre;
   float et;

   vec.Normalize();
   vec *= e;
   et = vec.Perp();

   return et;
}
}
