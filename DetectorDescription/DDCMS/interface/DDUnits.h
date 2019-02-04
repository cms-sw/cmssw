#ifndef DETECTOR_DESCRIPTION_DD_UNITS_H
#define DETECTOR_DESCRIPTION_DD_UNITS_H

#include <cmath>

#define ConvertTo(_x, _y) (_x)/(1.0_##_y)

constexpr long double _pi(M_PI);

constexpr long double operator"" _deg( long double deg )
{
    return deg*_pi/180;
}

constexpr long double operator"" _deg( unsigned long long int deg )
{
    return deg*_pi/180;
}

constexpr long double operator"" _mm( long double length )
{
    return length*1.;
}

constexpr long double operator"" _cm( long double length )
{
  return length*10.;
}

constexpr long double operator "" _pi( long double x ) 
{ return x * _pi; }

constexpr long double operator "" _pi( unsigned long long int x ) 
{ return x * _pi; } 

#endif
