#pragma GCC diagnostic ignored "-Wwrite-strings" //needed to get rid of pesky "deprecated conversion from string constant to char *" compilation error
#include "PhysicsTools/TagAndProbe/interface/RooCBExGaussShape.h"
#include "PhysicsTools/TagAndProbe/interface/ZGeneratorLineShape.h"
#include "PhysicsTools/TagAndProbe/interface/RooCMSShape.h"
#include "TVirtualFFT.h"

#ifdef __CINT__

//never even gets here...
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
//#pragma GCC diagnostic ignored "-Wformat"
// #pragma GCC diagnostic warning "-Wwrite-strings"

#pragma link C++ class RooCBExGaussShape;
#pragma link C++ class ZGeneratorLineShape;
#pragma link C++ class RooCMSShape;


#pragma link C++ global gROOT;
#pragma link C++ global gEnv;


#endif
