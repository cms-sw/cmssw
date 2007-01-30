#ifndef L1Trigger_TTPatternsGroup_h
#define L1Trigger_TTPatternsGroup_h
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
#include "L1Trigger/RPCTrigger/interface/TPatternsGroup.h"
 
class TTPatternsGroup: public TPatternsGroup {
    friend class RPCPacData;
    public:
      TTPatternsGroup();
};


#endif
