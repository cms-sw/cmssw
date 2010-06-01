//  COCOA class implementation file
//Id:  HistoDef.cc
//CAT: AnalysisMgr
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaFit/interface/HistoDef.h"
#include <math.h>

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
void HistoDef::init( ALIstring name)
{ 
  theName = name;
  theMin = ALI_DBL_MAX;
  theMax = -ALI_DBL_MAX;
}
