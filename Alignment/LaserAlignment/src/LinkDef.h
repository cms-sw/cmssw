
//#include "LASCommissioningData.h"
//#include "LASGlobalLoop.h"
#include "Alignment/LaserAlignment/interface/LASGlobalData.h"
#include "Alignment/LaserAlignment/interface/LASCoordinateSet.h"
#include "Alignment/LaserAlignment/interface/LASModuleProfile.h"

#include "TH1.h"
//#include "TDirectory.h"

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
//#pragma link C++ class LASCommissioningData+;  
//#pragma link C++ class LASGlobalLoop+;
#pragma link C++ class LASGlobalData<int>+;
#pragma link C++ class LASGlobalData<float>+;
#pragma link C++ class LASGlobalData<std::vector<float> >+;
#pragma link C++ class LASGlobalData<LASCoordinateSet>+;
#pragma link C++ class LASGlobalData<LASModuleProfile>+;
#pragma link C++ class LASGlobalData<std::pair<float, float> >+;
#pragma link C++ class LASGlobalData<unsigned int>+;
#pragma link C++ class LASGlobalData<std::string>+;
#pragma link C++ class LASGlobalData<TH1D*>+;
//#pragma link C++ class LASGlobalData<TDirectory*>+;
//#pragma link C++ class LASGlobalData<>+;
#endif
