#ifndef L1Trigger_RPCTEPatternsGroup_h
#define L1Trigger_RPCTEPatternsGroup_h
#include "L1Trigger/RPCTrigger/interface/TPatternsGroup.h"

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TEPatternsGroup
// 
/**
  \class TEPatternsGroup
  \brief Group of paterns for "improved"("energetic") algorithm.
  In current implementation all patterns in given group must have the same
  code and sign. All patterns must have the same m_QualityTabNumber.
  Patterns of given code and sign can be devided between a few EPatternsGroups,
  indexed by m_RefGroup.
  The group m_Code, m_Sign, m_RefGroup is definded by pattern index 0 in m_PatternsVec
 \author Karol Bunkowski (Warsaw),
 \author Tomasz Fruboes (Warsaw) - porting to CMSSW

*/

  class TEPatternsGroup: public TPatternsGroup {
    //friend class RPCPacData;
    friend class RPCPac;
    public:
      
      TEPatternsGroup(const RPCPattern::RPCPatVec::const_iterator& pattern);

    ///Checks, if patern can belong to this group, i.e. if has the same m_Code, m_Sign, m_RefGroup and m_QualityTabNumber.
      bool check(const RPCPattern::RPCPatVec::const_iterator& pattern);

    ///used for sorting TEPatternsGroups
      bool operator < (const TEPatternsGroup& ePatternsGroup) const;

    private:
      short m_QualityTabNumber;
  };



#endif
