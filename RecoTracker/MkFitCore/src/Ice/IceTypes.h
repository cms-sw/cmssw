//----------------------------------------------------------------------
/**
 *	Contains custom types.
 *	\file		IceTypes.h
 *	\author		Pierre Terdiman
 *	\date		April, 4, 2000
 */
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// Include Guard
#ifndef RecoTracker_MkFitCore_src_Ice_IceTypes_h
#define RecoTracker_MkFitCore_src_Ice_IceTypes_h

#include <cfloat>
#include <cstdlib>

#define inline_ inline

// Constants
const float PI = 3.14159265358979323846f;      //!< PI
const float HALFPI = 1.57079632679489661923f;  //!< 0.5 * PI
const float TWOPI = 6.28318530717958647692f;   //!< 2.0 * PI
const float INVPI = 0.31830988618379067154f;   //!< 1.0 / PI

const float RADTODEG = 57.2957795130823208768f;  //!< 180.0 / PI
const float DEGTORAD = 0.01745329251994329577f;  //!< PI / 180.0

const float EXP = 2.71828182845904523536f;      //!< e
const float INVLOG2 = 3.32192809488736234787f;  //!< 1.0 / log10(2)
const float LN2 = 0.693147180559945f;           //!< ln(2)
const float INVLN2 = 1.44269504089f;            //!< 1.0f / ln(2)

const float INV3 = 0.33333333333333333333f;    //!< 1/3
const float INV6 = 0.16666666666666666666f;    //!< 1/6
const float INV7 = 0.14285714285714285714f;    //!< 1/7
const float INV9 = 0.11111111111111111111f;    //!< 1/9
const float INV255 = 0.00392156862745098039f;  //!< 1/255

const float SQRT2 = 1.41421356237f;      //!< sqrt(2)
const float INVSQRT2 = 0.707106781188f;  //!< 1 / sqrt(2)

const float SQRT3 = 1.73205080757f;      //!< sqrt(3)
const float INVSQRT3 = 0.577350269189f;  //!< 1 / sqrt(3)

// Custom types used in ICE
typedef signed char sbyte;          //!< sizeof(sbyte)	must be 1
typedef unsigned char ubyte;        //!< sizeof(ubyte)	must be 1
typedef signed short sword;         //!< sizeof(sword)	must be 2
typedef unsigned short uword;       //!< sizeof(uword)	must be 2
typedef signed int sdword;          //!< sizeof(sdword)	must be 4
typedef unsigned int udword;        //!< sizeof(udword)	must be 4
typedef signed long long sqword;    //!< sizeof(sqword)	must be 8
typedef unsigned long long uqword;  //!< sizeof(uqword)	must be 8

// Added by M. Tadel (needed for 64-bit port)
typedef unsigned long sxword;  //!< pointer-sized   signed integer
typedef unsigned long uxword;  //!< pointer-sized unsigned integer

const udword OPC_INVALID_ID = 0xffffffff;  //!< Invalid dword ID (counterpart of 0 pointers)
const udword INVALID_NUMBER = 0xDEADBEEF;  //!< Standard junk value

// Type ranges
const sbyte MAX_SBYTE = 0x7f;          //!< max possible sbyte value
const sbyte MIN_SBYTE = 0x80;          //!< min possible sbyte value
const ubyte MAX_UBYTE = 0xff;          //!< max possible ubyte value
const ubyte MIN_UBYTE = 0x00;          //!< min possible ubyte value
const sword MAX_SWORD = 0x7fff;        //!< max possible sword value
const sword MIN_SWORD = 0x8000;        //!< min possible sword value
const uword MAX_UWORD = 0xffff;        //!< max possible uword value
const uword MIN_UWORD = 0x0000;        //!< min possible uword value
const sdword MAX_SDWORD = 0x7fffffff;  //!< max possible sdword value
const sdword MIN_SDWORD = 0x80000000;  //!< min possible sdword value
const udword MAX_UDWORD = 0xffffffff;  //!< max possible udword value
const udword MIN_UDWORD = 0x00000000;  //!< min possible udword value

const float MAX_FLOAT = FLT_MAX;                         //!< max possible float value
const float MIN_FLOAT = -FLT_MAX;                        //!< min possible loat value
const float ONE_OVER_RAND_MAX = 1.0f / float(RAND_MAX);  //!< Inverse of the max possible value returned by rand()

const udword IEEE_1_0 = 0x3f800000;        //!< integer representation of 1.0
const udword IEEE_255_0 = 0x437f0000;      //!< integer representation of 255.0
const udword IEEE_MAX_FLOAT = 0x7f7fffff;  //!< integer representation of MAX_FLOAT
const udword IEEE_MIN_FLOAT = 0xff7fffff;  //!< integer representation of MIN_FLOAT
const udword IEEE_UNDERFLOW_LIMIT = 0x1a000000;

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))                    //!< Returns the min value between a and b
#define MAX(a, b) ((a) > (b) ? (a) : (b))                    //!< Returns the max value between a and b
#define MAXMAX(a, b, c) ((a) > (b) ? MAX(a, c) : MAX(b, c))  //!< Returns the max value between a, b and c

template <class T>
inline_ const T& TMin(const T& a, const T& b) {
  return b < a ? b : a;
}
template <class T>
inline_ const T& TMax(const T& a, const T& b) {
  return a < b ? b : a;
}
template <class T>
inline_ void TSetMin(T& a, const T& b) {
  if (a > b)
    a = b;
}
template <class T>
inline_ void TSetMax(T& a, const T& b) {
  if (a < b)
    a = b;
}

#ifdef _WIN32
#define srand48(x) srand((unsigned int)(x))
#define srandom(x) srand((unsigned int)(x))
#define random() ((double)rand())
#define drand48() ((double)(((double)rand()) / ((double)RAND_MAX)))
#endif

#endif  // __ICETYPES_H__
