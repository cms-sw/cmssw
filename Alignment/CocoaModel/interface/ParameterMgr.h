//   COCOA class header file
//Id:  ParameterMgr.h
//CAT: Model
//
//   Manages the parameters of the input file (variables that are given a value to be reused in the file)
//
//   History: v1.0  11/11/01   Pedro Arce
#ifndef ParameterMgr_h
#define ParameterMgr_h

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <map>
typedef std::map<ALIstring, ALIdouble, std::less<ALIstring> > msd;

class ParameterMgr {
private:
  ParameterMgr(){};

public:
  static ParameterMgr* getInstance();
  ALIdouble getVal(const ALIstring& str, const ALIdouble dimensionFactor = 1.);

  void addParameter(const ALIstring& name, const ALIstring& valstr);
  void setRandomSeed(const long seed);
  void addRandomGaussParameter(const ALIstring& name, const ALIstring& valMean, const ALIstring& valStdDev);
  void addRandomFlatParameter(const ALIstring& name, const ALIstring& valMean, const ALIstring& valInterval);

  // get the parameter value if parameter name exists and return 1, else return 0
  ALIint getParameterValue(const ALIstring& name, ALIdouble& val);

private:
  static ParameterMgr* theInstance;

  msd theParameters;
};

#endif
