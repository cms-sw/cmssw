#ifndef __GLOBALS_HH
#define __GLOBALS_HH

//#include <ospace/std/string>
#include <string>
#include <cfloat>
#include <cmath>

typedef std::string ALIstring;

typedef long double ALIdouble;

typedef float ALIfloat;
 
typedef int ALIint;

typedef unsigned int ALIuint;

typedef bool ALIbool;

//const double ZERO = 1.E-50
const ALIdouble PI = M_PI; //2 * acos(0.0);

const double ALI_DBL_MAX = DBL_MAX;
const double ALI_DBL_MIN = 10./ALI_DBL_MAX;

namespace CLHEP{}

#endif
