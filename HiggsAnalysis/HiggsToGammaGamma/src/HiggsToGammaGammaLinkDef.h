// $Id: MitPhysicsUtilsLinkDef.h,v 1.12 2011/09/08 15:51:24 bendavid Exp $

#ifndef HIGGSANALYSIS_LINKDEF_H
#define HIGGSANALYSIS_LINKDEF_H


#include "../interface/GBRTree.h"
#include "../interface/GBRForest.h"



#endif

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;

#pragma link C++ class GBRTree+; 
#pragma link C++ class GBRForest+; 

#endif
