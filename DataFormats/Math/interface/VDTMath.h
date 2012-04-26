#ifndef vdt_math_h
#define vdt_math_h

/*******************************************************************************
 *
 * VDT math library: collection of double precision vectorisable trascendental
 * functions.
 * The c++11 standard is used: remember to enable it for the compilation.
 *
 * The basic idea is to exploit pade polinomials.
 * A lot of ideas were inspired by the cephes math library (by Stephen L. Moshier
 * moshier@na-net.ornl.gov) as well as actual code for the exp, log, sin, cos, 
 * tan, asin, acos and atan functions. The Cephes library can be found here:
 * http://www.netlib.org/cephes/
 * 
 ******************************************************************************/


#include <iostream>
#include <cmath>
#include <limits>

// used to enable some compiler option internally if needed
#define CMS_VECTORIZE
//#define CMS_VECTORIZE_VERBOSE __attribute__ ((optimize("-ftree-vectorizer-verbose=7")))
#define CMS_VECTORIZE_VERBOSE
//#define VDT_RESTRICT __restrict__
#define VDT_RESTRICT

#define VDT_FORCE_INLINE __attribute__((always_inline)) inline 

namespace vdt {

// paramters
constexpr double LOG2E = 1.4426950408889634073599; // 1/log(2)
constexpr double SQRTH = 0.70710678118654752440;

/*
Pade' polinomials coefficients and constants needed throughout the code.
The result of constexpr *here* is likely to be the same of const static
nevertheless we show its power: all the computations will take place at compile 
time
*/

// exp
constexpr double EXP_LIMIT = 708.;
constexpr double PX1exp = 1.26177193074810590878E-4;
constexpr double PX2exp = 3.02994407707441961300E-2;
constexpr double PX3exp = 9.99999999999999999910E-1;
constexpr double QX1exp = 3.00198505138664455042E-6;
constexpr double QX2exp = 2.52448340349684104192E-3;
constexpr double QX3exp = 2.27265548208155028766E-1;
constexpr double QX4exp = 2.00000000000000000009E0;

// logarithm
constexpr double LOG_UPPER_LIMIT = 1e307;
constexpr double LOG_LOWER_LIMIT = 1e-307;
constexpr double PX1log = 1.01875663804580931796E-4;
constexpr double PX2log = 4.97494994976747001425E-1;
constexpr double PX3log = 4.70579119878881725854E0;
constexpr double PX4log = 1.44989225341610930846E1;
constexpr double PX5log = 1.79368678507819816313E1;
constexpr double PX6log = 7.70838733755885391666E0;

constexpr double QX1log = 1.12873587189167450590E1;
constexpr double QX2log = 4.52279145837532221105E1;
constexpr double QX3log = 8.29875266912776603211E1;
constexpr double QX4log = 7.11544750618563894466E1;
constexpr double QX5log = 2.31251620126765340583E1;

// Sin and cos
constexpr double DP1sc = 7.85398125648498535156E-1;
constexpr double DP2sc = 3.77489470793079817668E-8;
constexpr double DP3sc = 2.69515142907905952645E-15;
constexpr double TWOPI = 2.*M_PI;
constexpr double PI = M_PI;
constexpr double PIO2 = M_PI_2;
constexpr double PIO4 = M_PI_4;

// Sin
constexpr double SIN_UPPER_LIMIT = TWOPI;
constexpr double SIN_LOWER_LIMIT = -SIN_UPPER_LIMIT;
constexpr double C1sin = 1.58962301576546568060E-10;
constexpr double C2sin =-2.50507477628578072866E-8;
constexpr double C3sin = 2.75573136213857245213E-6;
constexpr double C4sin =-1.98412698295895385996E-4;
constexpr double C5sin = 8.33333333332211858878E-3;
constexpr double C6sin =-1.66666666666666307295E-1;

//Cos
constexpr double C1cos =-1.13585365213876817300E-11;
constexpr double C2cos = 2.08757008419747316778E-9;
constexpr double C3cos =-2.75573141792967388112E-7;
constexpr double C4cos = 2.48015872888517045348E-5;
constexpr double C5cos =-1.38888888888730564116E-3;
constexpr double C6cos = 4.16666666666665929218E-2;

// Asin and acos

constexpr double RX1asin = 2.967721961301243206100E-3;
constexpr double RX2asin = -5.634242780008963776856E-1;
constexpr double RX3asin = 6.968710824104713396794E0;
constexpr double RX4asin = -2.556901049652824852289E1;
constexpr double RX5asin = 2.853665548261061424989E1;

constexpr double SX1asin = -2.194779531642920639778E1;
constexpr double SX2asin =  1.470656354026814941758E2;
constexpr double SX3asin = -3.838770957603691357202E2;
constexpr double SX4asin = 3.424398657913078477438E2;

constexpr double PX1asin = 4.253011369004428248960E-3;
constexpr double PX2asin = -6.019598008014123785661E-1;
constexpr double PX3asin = 5.444622390564711410273E0;
constexpr double PX4asin = -1.626247967210700244449E1;
constexpr double PX5asin = 1.956261983317594739197E1;
constexpr double PX6asin = -8.198089802484824371615E0;

constexpr double QX1asin = -1.474091372988853791896E1;
constexpr double QX2asin =  7.049610280856842141659E1;
constexpr double QX3asin = -1.471791292232726029859E2;
constexpr double QX4asin = 1.395105614657485689735E2;
constexpr double QX5asin = -4.918853881490881290097E1;

//Tan
constexpr double PX1tan=-1.30936939181383777646E4;
constexpr double PX2tan=1.15351664838587416140E6;
constexpr double PX3tan=-1.79565251976484877988E7;

constexpr double QX1tan = 1.36812963470692954678E4;
constexpr double QX2tan = -1.32089234440210967447E6;
constexpr double QX3tan = 2.50083801823357915839E7;
constexpr double QX4tan = -5.38695755929454629881E7;

constexpr double DP1tan = 7.853981554508209228515625E-1;
constexpr double DP2tan = 7.94662735614792836714E-9;
constexpr double DP3tan = 3.06161699786838294307E-17;
constexpr double TAN_LIMIT = TWOPI;

// Atan
constexpr double T3PO8 = 2.41421356237309504880;
constexpr double MOREBITS = 6.123233995736765886130E-17;
constexpr double MOREBITSO2 = MOREBITS/2.;

constexpr double PX1atan = -8.750608600031904122785E-1;
constexpr double PX2atan = -1.615753718733365076637E1;
constexpr double PX3atan = -7.500855792314704667340E1;
constexpr double PX4atan = -1.228866684490136173410E2;
constexpr double PX5atan = -6.485021904942025371773E1;

constexpr double QX1atan = - 2.485846490142306297962E1;
constexpr double QX2atan = 1.650270098316988542046E2;
constexpr double QX3atan = 4.328810604912902668951E2;
constexpr double QX4atan = 4.853903996359136964868E2;
constexpr double QX5atan = 1.945506571482613964425E2;

constexpr double ATAN_LIMIT = 1e307;

// Inverse Sqrt
// constexpr unsigned int ISQRT_ITERATIONS = 4;
constexpr double SQRT_LIMIT = 1e307;

// Service----------------------------------------------------------------------
/// Print the instructions used on screen
void print_instructions_info();

//------------------------------------------------------------------------------

/// Used to switch between different type of interpretations of the data (64 bits)
typedef union {
  double d;
  int i[2];
  long long ll;
  unsigned short s[4];
} ieee754;

//------------------------------------------------------------------------------

/// Converts an unsigned long long to a double
VDT_FORCE_INLINE double ll2d(unsigned long long x) {
  ieee754 tmp;
  tmp.ll=x;
  return tmp.d;
}

//------------------------------------------------------------------------------

/// Converts a double to an unsigned long long
VDT_FORCE_INLINE unsigned long long d2ll(double x) {
  ieee754 tmp;
  tmp.d=x;
  return tmp.ll;
}

//------------------------------------------------------------------------------
/// Like frexp but vectorising and the exponent is a double.
VDT_FORCE_INLINE double getMantExponent(double x, double& fe){
  
  unsigned long long n = d2ll(x);
  
  // shift to the right up to the beginning of the exponent 
  // then with a mask, cut off the sign bit
  unsigned long long le = ((n >> 52) & 0x7ffLL);
  
  // chop the head of the number: an int contains more than 11 bits (32)
  int e = le; // This is important since sums on ull do not vectorise
  fe = (e-1023) +1 ; // the plus one to make the result identical to frexp
  
  // 13 times f means 52 1. Masking with this means putting to 0 exponent
  // and sign of a double, leaving the Mantissa, the first 52 bits of a double.
  n &=0xfffffffffffffLL;
  
  // build a mask which is 0.5, i.e. an exponent equal to 1022
  // which means *2, see the above +1.
  const unsigned long long p05 = d2ll(0.5);
  n |= p05;
  x = ll2d(n);
  return x;
}

//------------------------------------------------------------------------------
// Now the mathematical functions are encoded.

// Exp -------------------------------------------------------------------------
// Vectorises in a loop without any change in 4.7
/// Exponential Function
VDT_FORCE_INLINE double fast_exp(double x){

    double initial_x = x;

//    double px =int(LOG2E * x + 0.5); // std::floor(LOG2E * x + 0.5);
    double px = std::floor(LOG2E * x + 0.5);

    int n = px;

    x -= px * 6.93145751953125E-1;
    x -= px * 1.42860682030941723212E-6;

    double xx = x * x;

    // px = x * P(x**2).
    px = PX1exp;
    px *= xx;
    px += PX2exp;
    px *= xx;
    px += PX3exp;
    px *= x;

    // Evaluate Q(x**2).
    double qx = QX1exp;
    qx *= xx;
    qx += QX2exp;
    qx *= xx;
    qx += QX3exp;
    qx *= xx;
    qx += QX4exp;

    // e**x = 1 + 2x P(x**2)/( Q(x**2) - P(x**2) )
    x = px / (qx - px);
    x = 1.0 + 2.0 * x;

    // Build 2^n in double.
    ieee754 u;
    u.d = 0;
    n += 1023;
    u.ll = (long long) (n) << 52;

    double res = x * u.d;
    if (initial_x > EXP_LIMIT)
            res = std::numeric_limits<double>::infinity();
    if (initial_x < -EXP_LIMIT)
            res = 0.;   
    
    return res;
}


// Log -------------------------------------------------------------------------

VDT_FORCE_INLINE double fast_log(double x){

    double input_x=x;

    /* separate mantissa from exponent */    
    double fe;
    x = getMantExponent(x,fe);

    // blending      
    if( x < SQRTH ) {
      fe-=1;
      x +=  x ;
      }
    x -= 1.0;

    /* rational form */

    double z = x*x;
    double px =  PX1log;
    px *= x;    
    px += PX2log;
    px *= x;    
    px += PX3log;
    px *= x; 
    px += PX4log;
    px *= x; 
    px += PX5log;
    px *= x;
    px += PX6log;
    //
    //for the final formula
    px *= x; 
    px *= z;

    double qx = x;
    qx += QX1log;
    qx *=x;
    qx += QX2log;
    qx *=x;    
    qx += QX3log;
    qx *=x;    
    qx += QX4log;
    qx *=x;    
    qx += QX5log;

    double y = px / qx ;

    y -= fe * 2.121944400546905827679e-4; 
    y -= 0.5 * z  ;

    z = x + y;
    z += fe * 0.693359375;

    if (input_x > LOG_UPPER_LIMIT)
      z = std::numeric_limits<double>::infinity();
    if (input_x < LOG_LOWER_LIMIT)
      z =  - std::numeric_limits<double>::infinity();       

//     std::cout << input_x << " " << std::log(input_x) << " " << z << std::endl;

    return( z );  
  
  }

//------------------------------------------------------------------------------
/// Sin defined between -2pi and 2pi
VDT_FORCE_INLINE double fast_sin(double x){

  int sign = 1;

  if (x < 0){
    x = - x;
    sign = -1;
    }

  if( x > PI ){
    x = TWOPI - x;
    sign = - sign;
    }

  if( x > PIO2 )
    x = PI - x ;

  double y = int( x/PIO4 ); // integer part of x/PIO4 

  int j=0;
  if (x>PIO4){
     j=2;
     y+=1;
     }

  /* Extended precision modular arithmetic */
  double z = ((x - y * DP1sc) - y * DP2sc) - y * DP3sc;

  double zz = z * z;

  double px=0;

  if( j==2 ){
    px  = C1cos;
    px *= zz;
    px += C2cos;
    px *= zz;
    px += C3cos;
    px *= zz;
    px += C4cos;
    px *= zz;
    px += C5cos;
    px *= zz;
    px += C6cos;
    y = 1.0 - zz * .5 + zz * zz * px;
    }
  else{
    px  = C1sin;
    px *= zz;
    px += C2sin;
    px *= zz;
    px += C3sin;
    px *= zz;
    px += C4sin;
    px *= zz;
    px += C5sin;
    px *= zz;
    px += C6sin;
    y = z  +  z * zz * px;
    }

  y *= sign;

  return y;
  }

//------------------------------------------------------------------------------

VDT_FORCE_INLINE double fast_asin(double x){

  int sign=1;
  double a = x; //necessary for linear approx

  if ( x < 0. ){
    sign *= -1;
    a *= -1;
    }

  double p, z, zz;
  double px,qx;

  /* arcsin(1-x) = pi/2 - sqrt(2x)(1+R(x))  */
  zz = 1.0 - a;
  
  px = RX1asin;
  px*= zz;
  px+= RX2asin;
  px*= zz;    
  px+= RX3asin;
  px*= zz;    
  px+= RX4asin;
  px*= zz;    
  px+= RX5asin;

  qx = zz;
  qx+= SX1asin;
  qx*= zz;    
  qx+= SX2asin;
  qx*= zz;    
  qx+= SX3asin;
  qx*= zz;    
  qx+= SX4asin;
  
  p =zz* px/qx;
  
//     p = zz * polevl( zz, R, 4)/p1evl( zz, S, 4);
  
  zz = std::sqrt(zz+zz);
  z = PIO4 - zz;
  zz = zz * p - MOREBITS;
  z -= zz;
  z += PIO4;
    
  if( a < 0.625 ){
    zz = a * a;
    px = PX1asin;
    px*= zz;
    px+= PX2asin;
    px*= zz;    
    px+= PX3asin;
    px*= zz;    
    px+= PX4asin;
    px*= zz;    
    px+= PX5asin;
    px*= zz;    
    px+= PX6asin;

    qx = zz;
    qx+= QX1asin;
    qx*= zz;    
    qx+= QX2asin;
    qx*= zz;    
    qx+= QX3asin;
    qx*= zz;    
    qx+= QX4asin;
    qx*= zz;    
    qx+= QX5asin;
    
    z = zz*px/qx;    

    z = a * z + a;
    }

  z *= sign;

   //linear approx, not sooo needed but seable. Price is cheap though
  if( a < 1.0e-8 )
    z = a;
  
  return z;
  }

//------------------------------------------------------------------------------

    
/// Cos defined between -2pi and 2pi
VDT_FORCE_INLINE double fast_cos(double x){

  x = std::abs(x);

  if( x > PI )
    x = TWOPI - x ;

  int sign = 1;
   if( x > PIO2 ){
     x = PI - x;
     sign=-1;
     }

  double y = int( x/PIO4 ); // integer part of x/PIO4 

  int j=0;
  if (x>PIO4){
    j=2;
    y+=1;
    sign = -sign;
    }

  /* Extended precision modular arithmetic */
  double z = ((x - y * DP1sc) - y * DP2sc) - y * DP3sc;

  double zz = z * z;

  double px=0;
  if( j==2 ){
    px  = C1sin;
    px *= zz;
    px += C2sin;
    px *= zz;
    px += C3sin;
    px *= zz;
    px += C4sin;
    px *= zz;
    px += C5sin;
    px *= zz;
    px += C6sin;
    y = z  +  z * zz * px;
    }
  else{
    px  = C1cos;
    px *= zz;
    px += C2cos;
    px *= zz;
    px += C3cos;
    px *= zz;
    px += C4cos;
    px *= zz;
    px += C5cos;
    px *= zz;
    px += C6cos;  
    y = 1. - zz * .5 + zz * zz * px;  
    }
  
  y *= sign;
  
  return y;
  }

// Acos ------------------------------------------------------------------------
VDT_FORCE_INLINE double fast_acos(double x){
  double z;
 
//   z = PIO4 - fast_asin(x);
//   z += MOREBITS;
//   z += PIO4;

//   if (x > .5 )
    z = 2.0 * fast_asin(  sqrt(0.5 - 0.5 * x ) ) ;
    
  return z;
  }

// Tangent  --------------------------------------------------------------------
/// Sin defined between -2pi and 2pi
VDT_FORCE_INLINE double fast_tan( double x ){
/* DP
 * Some of the ifs had to be skipped and replaced by calculations. This allowed 
 * the vectorisation but introduced a loss of performance. 
 * A solution should be found
*/  
  
   // make argument positive but save the sign
   // without ifs
   double abs_x =std::abs(x);
   int sign = x/abs_x;
   x = abs_x;

// remove this if
//     if (x > PI)
//        x = x - PI;
// like this:
   int nPI  = x /PI;
   x = x - nPI * PI;


//     reflect and flip with if
//     if (x > PIO2){
//        x = PI - x ;
//        sign = - sign;
//       }
// and without
    int nPIO2 = x/PIO2;
    int factor = ( 1 - 2* nPIO2);
    x = nPIO2* PI + factor * x;
    sign *= factor;


    /* compute x mod PIO4 */
    int nPIO4 = x/PIO4;
    double y = 2 * nPIO4;

    /* integer and fractional part modulo one octant */

//     This if can be removed and the expression becomes
//     if (x > PIO4){
//        y=2.0;
//       }
//   like this:
//     y = y  + nPIO4;


    double z = ((x - y * DP1tan) - y * DP2tan) - y * DP3tan;

    double zz = z * z;

    y=z;

    if( zz > 1.0e-14 ){
        double px = PX1tan;
        px *= zz;
        px += PX2tan;
        px *= zz;
        px += PX3tan;

        double qx=zz;
        qx += QX1tan;
        qx *=zz;
        qx += QX2tan;
        qx *=zz;
        qx += QX3tan;
        qx *=zz;
        qx += QX4tan;

        y = z + z * zz * px / qx;
    }

    // here if we are in the second octant we have
    //  y = -1 /y
   // else we have y!
   // again a trick not to use ifs...
    y -= nPIO4 * (  y  + 1.0 / y);

    y *= sign;

    return y ;
    }

// Atan -------------------------------------------------------------------------
// REMEMBER pi/2 == inf!!
VDT_FORCE_INLINE double fast_atan(double x){

    /* make argument positive and save the sign */
    int sign = 1;
    if( x < 0.0 ) {
        x = - x;
        sign = -1;
        }

    /* range reduction */
    double originalx=x;

// This is slower!
//     double y = 0.0;
//     double factor = 0.;
// 
//     if (x  > .66){
//         y = PIO4;
//         factor = MOREBITSO2;
//         x = (x-1.0) / (x+1.0);
//         }
//     if( originalx > T3PO8 ) {
//         y = PIO2;
//         factor = MOREBITS;
//         x = -1.0 / originalx ;
//         }

    double y = PIO4;
    double factor = MOREBITSO2;
    x = (x-1.0) / (x+1.0);

    if( originalx > T3PO8 ) {
        y = PIO2;
        //flag = 1.;
        factor = MOREBITS;
        x = -1.0 / originalx ;
        }
    if ( originalx <= 0.66 ) {
        y = 0.0;
        x = originalx;
        //flag = 0.;
        factor = 0.;
        }

    double z = x * x;

    double px = PX1atan;
    px *= z;
    px += PX2atan;
    px *= z;
    px += PX3atan;
    px *= z;
    px += PX4atan;
    px *= z;
    px += PX5atan;
    px *= z; // for the final formula

    double qx=z;
    qx += QX1atan;
    qx *=z;
    qx += QX2atan;
    qx *=z;
    qx += QX3atan;
    qx *=z;
    qx += QX4atan;
    qx *=z;
    qx += QX5atan;

//     z = px / qx;
//     z = x * px / qx + x;

    y = y +x * px / qx + x +factor;

    y = sign * y;

    return y;
    }

//------------------------------------------------------------------------------

// Taken from from quake and remixed :-)

VDT_FORCE_INLINE double fast_isqrt_general(double x, const unsigned short ISQRT_ITERATIONS) { 

  double x2 = x * 0.5;
  double y  = x;
  unsigned long long i  = d2ll(y);
  // Evil!
  i  = 0x5fe6eb50c7aa19f9  - ( i >> 1 );
  y  = ll2d(i);
  for (unsigned int j=0;j<ISQRT_ITERATIONS;++j)
      y *= 1.5 - ( x2 * y * y ) ;

  return y;
}



//------------------------------------------------------------------------------

// Four iterations
VDT_FORCE_INLINE double fast_isqrt(double x) {return fast_isqrt_general(x,4);} 

// Two iterations
VDT_FORCE_INLINE double fast_approx_isqrt(double x) {return fast_isqrt_general(x,3);}

//------------------------------------------------------------------------------

VDT_FORCE_INLINE double std_isqrt (double x) {return 1./std::sqrt(x);}

//------------------------------------------------------------------------------

VDT_FORCE_INLINE double fast_inv (double x) {
    double sign = 1;
    if( x < 0.0 ) {
        x = - x;
        sign = -1;
        }
    double y=fast_isqrt(x); 
    return y*y*sign;
    }

VDT_FORCE_INLINE double fast_approx_inv (double x) {
    double sign = 1;
    if( x < 0.0 ) {
        x = - x;
        sign = -1;
        }
    double y=fast_approx_isqrt(x); 
    return y*y*sign;}

VDT_FORCE_INLINE double std_inv (double x) {return 1./x;}

//------------------------------------------------------------------------------
// Some preprocessor in order to avoid a lot of error prone repetitions
// CMS_VECTORIZE_VERBOSE is a preprocessor variable in a preprocessor function

// Fast vector functions 
#define FAST_VECT_FUNC(NAME) __attribute__((always_inline)) inline void NAME##_vect(double const * VDT_RESTRICT input , double * VDT_RESTRICT outupt, const unsigned int arr_size) { \
  for (unsigned int i=0;i<arr_size;++i) \
    outupt[i] = NAME ( input[i] ) CMS_VECTORIZE_VERBOSE; \
    }

/// Some tweaks to make it vectorise with gcc46
void fast_exp_vect_46(double const* input, double* output, const unsigned int arr_size);

// Profitability threshold = 3
FAST_VECT_FUNC(fast_exp)

/// Some tweaks to make it vectorise with gcc46
void fast_log_vect_46(double const* input, double* output, const unsigned int arr_size);

// Profitability threshold = 3
FAST_VECT_FUNC(fast_log)

// Profitability threshold = 7
FAST_VECT_FUNC(fast_sin)

// Profitability threshold = 3
FAST_VECT_FUNC(fast_asin)

// Profitability threshold = 7
FAST_VECT_FUNC(fast_cos)

// Profitability threshold = 3
FAST_VECT_FUNC(fast_acos)

//Profitability threshold = 3
FAST_VECT_FUNC(fast_tan)

//Profitability threshold = 3
FAST_VECT_FUNC(fast_atan)
  
//Profitability threshold = 2 (2!!!)
FAST_VECT_FUNC(fast_isqrt)

//Profitability threshold = 2 (2!!!)
FAST_VECT_FUNC(fast_approx_isqrt)

//Profitability threshold = 2
FAST_VECT_FUNC(fast_inv)

//Profitability threshold = 2
FAST_VECT_FUNC(fast_approx_inv)

//------------------------------------------------------------------------------
// Reference vector functions
#define VECT_FUNC(NAME) __attribute__((always_inline)) inline void std_##NAME##_vect(double const * VDT_RESTRICT input , double* VDT_RESTRICT outupt, const unsigned int arr_size) { \
  for (unsigned int i=0;i<arr_size;++i) \
    outupt[i] = std::NAME ( input[i] ) CMS_VECTORIZE_VERBOSE; \
    }

VECT_FUNC(exp)

VECT_FUNC(log)

VECT_FUNC(sin)

VECT_FUNC(asin)

VECT_FUNC(cos)

VECT_FUNC(acos)

VECT_FUNC(tan)

VECT_FUNC(atan)

VDT_FORCE_INLINE void std_isqrt_vect(double const * VDT_RESTRICT input , 
                    double* VDT_RESTRICT output, 
                    const unsigned int arr_size) CMS_VECTORIZE_VERBOSE{
  //Profitability threshold = 6
  for (unsigned int i=0;i<arr_size;++i)
    output[i] = vdt::std_isqrt(input[i]);
  }

VDT_FORCE_INLINE void std_inv_vect(double const * VDT_RESTRICT input , 
                    double* VDT_RESTRICT output, 
                    const unsigned int arr_size) CMS_VECTORIZE_VERBOSE{
  //Profitability threshold = 6
  for (unsigned int i=0;i<arr_size;++i)
    output[i] = vdt::std_inv(input[i]);
  }


//------------------------------------------------------------------------------

} // end of vdt namespace

#endif

