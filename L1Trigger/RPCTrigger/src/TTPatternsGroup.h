#ifndef RPCTrigger_TTPatternsGroup_h
#define RPCTrigger_TTPatternsGroup_h
// -*- C++ -*-
//
// Package:     RPCTrigger
// Class  :     TTPatternsGroup
// 
/**
 \class TTPatternsGroup TTPatternsGroup.h L1Trigger/RPCTrigger/interface/TTPatternsGroup.h
 \brief Group of paterns, for which the "baseline"("track") algorithm is performed. 
 \author Karol Bunkowski (Warsaw),
 \author Tomasz Fruboes (Warsaw) - porting to CMSSW

*/
#include "L1Trigger/RPCTrigger/src/TPatternsGroup.h"
 
class TTPatternsGroup: public TPatternsGroup {
    friend class RPCPac;
    public:
      TTPatternsGroup();
};


#endif
