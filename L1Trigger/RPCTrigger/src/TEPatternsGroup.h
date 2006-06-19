#ifndef L1Trigger_TEPatternsGroup_h
#define L1Trigger_TEPatternsGroup_h
#include "L1Trigger/RPCTrigger/src/TPatternsGroup.h"

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TEPatternsGroup
// 
/**
  \class TEPatternsGroup
  \brief Group of paterns for "improved"("energetic") algorithm.
  In current implementation all patterns in given group must have the same
  code and sign. All patterns must have the same QualityTabNumber.
  Patterns of given code and sign can be devided between a few EPatternsGroups,
  indexed by RefGroup.
  The group Code, Sign, RefGroup is definded by pattern index 0 in PatternsVec
 \author Karol Bunkowski (Warsaw),
 \author Tomasz Fruboes (Warsaw) - porting to CMSSW

*/

  class TEPatternsGroup: public TPatternsGroup {
    friend class L1RpcPac;
    public:
      
      TEPatternsGroup(const L1RpcPatternsVec::const_iterator& pattern);

    ///Checks, if patern can belong to this group, i.e. if has the same Code, Sign, RefGroup and QualityTabNumber.
      bool Check(const L1RpcPatternsVec::const_iterator& pattern);

    ///used for sorting TEPatternsGroups
      bool operator < (const TEPatternsGroup& ePatternsGroup) const;

    private:
      short QualityTabNumber;
  };



#endif
