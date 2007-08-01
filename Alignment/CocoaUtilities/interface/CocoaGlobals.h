#ifndef __GLOBALS_HH
#define __GLOBALS_HH

//#include <ospace/std/string>
#include <string>

typedef std::string ALIstring;

typedef long double ALIdouble;

typedef float ALIfloat;
 
typedef int ALIint;

typedef unsigned int ALIuint;

typedef bool ALIbool;

//const double ZERO = 1.E-50
const ALIdouble PI = 3.1415926;

const double ALI_DBL_MAX = 1.E99;
const double ALI_DBL_MIN = -1.E99;

namespace CLHEP{}
using namespace CLHEP;

#endif
