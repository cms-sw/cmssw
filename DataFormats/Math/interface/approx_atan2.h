#ifndef APPROX_ATAN2_H
#define APPROX_ATAN2_H


/*
 * approximate atan2 evaluations
 *
 * Polynomials were obtained using Sollya scripts (in comments below)
 *
 *
*/

/*
f= atan((1-x)/(1+x))-atan(1);
I=[-1+10^(-4);1.0];
filename="atan.txt";
print("") > filename;
for deg from 3 to 11 do begin
  p = fpminimax(f, deg,[|1,23...|],I, floating, absolute); 
  display=decimal;
  acc=floor(-log2(sup(supnorm(p, f, I, absolute, 2^(-20)))));
  print( "   // degree = ", deg, 
         "  => absolute accuracy is ",  acc, "bits" ) >> filename;
  print("template<> inline float approx_atan2f_P<", deg, ">(float x){") >> filename;
  display=hexadecimal;
  print(" return ", horner(p) , ";") >> filename;
  print("}") >> filename;
end;
*/




#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>


// float 

template<int DEGREE>
inline float approx_atan2f_P(float x);

// degree =  3   => absolute accuracy is  7 bits
template<> inline float approx_atan2f_P< 3 >(float x){
  return x * (float(-0xf.8eed2p-4) + x*x * float(0x3.1238p-4)) ;
}

// degree =  5   => absolute accuracy is  10 bits
template<> inline float approx_atan2f_P< 5 >(float x){
  auto z = x*x;
  return  x * (float(-0xf.ecfc8p-4) + z * (float(0x4.9e79dp-4) + z * float(-0x1.44f924p-4) ) );
}

// degree =  7   => absolute accuracy is  13 bits
template<> inline float approx_atan2f_P< 7 >(float x){
  auto z = x*x;
  return  x * (float(-0xf.fcc7ap-4) + z * (float(0x5.23886p-4) + z * (float(-0x2.571968p-4) + z * float(0x9.fb05p-8) ) ) ) ;
}

// degree =  9   => absolute accuracy is  16 bits
template<> inline float approx_atan2f_P< 9 >(float x){
  auto z = x*x;
  return x * (float(-0xf.ff73ep-4) + z * (float(0x5.48ee1p-4) + z * (float(-0x2.e1efe8p-4) + z * (float(0x1.5cce54p-4) + z * float(-0x5.56245p-8) ) ) ) ); 
}

// degree =  11   => absolute accuracy is  19 bits
template<> inline float approx_atan2f_P< 11 >(float x){
  auto z = x*x;
  return x * (float(-0xf.ffe82p-4) + z * (float(0x5.526c8p-4) + z * (float(-0x3.18bea8p-4) + z * (float(0x1.dce3bcp-4) + z * (float(-0xd.7a64ap-8) + z * float(0x3.000eap-8))))));
}

// degree =  13   => absolute accuracy is  21 bits
template<> inline float approx_atan2f_P< 13 >(float x){
  auto z = x*x;
  return  x * (float(-0xf.fffbep-4) + z * (float(0x5.54adp-4) + z * (float(-0x3.2b4df8p-4) + z * (float(0x2.1df79p-4) + z * (float(-0x1.46081p-4) + z * (float(0x8.99028p-8) + z * float(-0x1.be0bc4p-8))))))) ;
}

// degree =  15   => absolute accuracy is  24 bits
template<> inline float approx_atan2f_P< 15 >(float x){
  auto z = x*x;
  return x * (float(-0xf.ffff4p-4) + z * (float(0x5.552f9p-4 + z * (float(-0x3.30f728p-4) + z * (float(0x2.39826p-4) + z * (float(-0x1.8a880cp-4) + z * (float(0xe.484d6p-8) + z * (float(-0x5.93d5p-8) + z * float(0x1.0875dcp-8)))))))));
}


template<int DEGREE>
inline float unsafe_atan2f_impl(float y, float x) {

  constexpr float pi4f =  3.1415926535897932384626434/4;
  constexpr float pi34f = 3.1415926535897932384626434*3/4;

  auto r= (std::abs(x) - std::abs(y))/(std::abs(x) + std::abs(y));
  if (x<0) r = -r;

  auto angle = (x>=0) ? pi4f : pi34f;
  angle += approx_atan2f_P<DEGREE>(r);
  

  return ( (y < 0)) ? - angle : angle ;


}

template<int DEGREE>
inline float unsafe_atan2f(float y, float x) {
  return unsafe_atan2f_impl<DEGREE>(y,x);

}

template<int DEGREE>
inline float safe_atan2f(float y, float x) {
  return unsafe_atan2f_impl<DEGREE>( y, (y==0.f)&(x==0.f) ? 0.2f :  x);
  // return (y==0.f)&(x==0.f) ? 0.f :  unsafe_atan2f_impl<DEGREE>( y, x);

}


// integer...
/*
  f= (2^31/pi)*(atan((1-x)/(1+x))-atan(1));
  I=[-1+10^(-4);1.0];
  p = fpminimax(f, [|1,3,5,7,9,11|],[|23...|],I, floating, absolute);
 */


template<int DEGREE>
inline float approx_atan2i_P(float x);

// degree =  3   => absolute accuracy is  6*10^6
template<> inline float approx_atan2i_P< 3 >(float x){
 auto z = x*x;
 return x * (-664694912.f + z * 131209024.f);
}

// degree =  5   => absolute accuracy is  4*10^5
template<> inline float approx_atan2i_P< 5 >(float x){
 auto z = x*x;
 return x * (-680392064.f + z * (197338400.f + z * (-54233256.f)));
}

// degree =  7   => absolute accuracy is  6*10^4
template<> inline float approx_atan2i_P< 7 >(float x){
 auto z = x*x;
 return  x * (-683027840.f + z * (219543904.f + z * (-99981040.f + z * 26649684.f)));
}

// degree =  9   => absolute accuracy is  8000
template<> inline float approx_atan2i_P< 9 >(float x){
 auto z = x*x;
 return   x * (-683473920.f + z * (225785056.f + z * (-123151184.f + z * (58210592.f + z * (-14249276.f)))));
}

// degree =  11   => absolute accuracy is  1000
template<> inline float approx_atan2i_P< 11 >(float x){
 auto z = x*x;
 return   x * (-683549696.f + z * (227369312.f + z * (-132297008.f + z * (79584144.f + z * (-35987016.f + z * 8010488.f)))));
}

// degree =  13   => absolute accuracy is  163
template<> inline float approx_atan2i_P< 13 >(float x){
 auto z = x*x;
 return  x * (-683562624.f + z * (227746080.f + z * (-135400128.f + z * (90460848.f + z * (-54431464.f + z * (22973256.f + z * (-4657049.f)))))));
}

template<> inline float approx_atan2i_P< 15 >(float x){
 auto z = x*x;
 return  x * (-683562624.f + z * (227746080.f + z * (-135400128.f + z * (90460848.f + z * (-54431464.f + z * (22973256.f + z * (-4657049.f)))))));
}


template<int DEGREE>
inline int unsafe_atan2i_impl(float y, float x) {


  constexpr long long maxint = (long long)(std::numeric_limits<int>::max())+1LL;
  constexpr int pi4 =  int(maxint/4LL);
  constexpr int pi34 = int(3LL*maxint/4LL);

  auto r= (std::abs(x) - std::abs(y))/(std::abs(x) + std::abs(y));
  if (x<0) r = -r;

  auto angle = (x>=0) ? pi4 : pi34;
  angle += int(approx_atan2i_P<DEGREE>(r));
  // angle += int(std::round(approx_atan2i_P<DEGREE>(r)));


  return (y < 0) ? - angle : angle ;


}

template<int DEGREE>
inline int unsafe_atan2i(float y, float x) {
  return unsafe_atan2i_impl<DEGREE>(y,x);

}

inline
int phi2int(float x) {
  constexpr float p2i = ( (long long)(std::numeric_limits<int>::max())+1LL )/M_PI;
  return std::round(x*p2i);
}

inline
float int2phi(int x) {
  constexpr float i2p = M_PI/( (long long)(std::numeric_limits<int>::max())+1LL );
  return float(x)*i2p;
}

inline
double int2dphi(int x) {
  constexpr double i2p = M_PI/( (long long)(std::numeric_limits<int>::max())+1LL );
  return x*i2p;
}


inline
short phi2short(float x) {
  constexpr float p2i = ( (int)(std::numeric_limits<short>::max())+1 )/M_PI;
  return std::round(x*p2i);
}

inline
float short2phi(short x) {
  constexpr float i2p = M_PI/( (int)(std::numeric_limits<short>::max())+1 );
  return float(x)*i2p;
}


#endif


