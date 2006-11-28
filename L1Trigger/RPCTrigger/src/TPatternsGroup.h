#ifndef L1Trigger_TPatternsGroup_h
#define L1Trigger_TPatternsGroup_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TPatternsGroup
// 
/**\class TPatternsGroup TPatternsGroup.h src/L1Trigger/interface/TPatternsGroup.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
#include "L1Trigger/RPCTrigger/src/RPCPattern.h"
#include "L1Trigger/RPCTrigger/src/RPCLogCone.h"
  /** \class TPatternsGroup
 * Basic class for storing grouped patterns inside Pac.
 * In group (object of class TPatternsGroup) the patterns belonging to given
   * group are stored in m_PatternsVec. These patterns are use in trigger algorithm*/
  class TPatternsGroup {
    friend class RPCPac;
    protected:
      RPCConst::TPatternType m_PatternsGroupType;
    //L1RpcPatternsVec m_PatternsVec; //!< Vector of patterns.
      std::vector<L1RpcPatternsVec::const_iterator> m_PatternsItVec; //!< Vector of itereator on m_PatternsVec in Pac.
      RPCLogCone m_GroupShape; //!< Set LogStrips denotes strips beloging to the group.
      std::string m_GroupDescription;

    public:

      void addPattern(const L1RpcPatternsVec::const_iterator& pattern);

    ///Updates m_GroupShape, i.e. sets to true strips belonging to the pattern. Coleed in addPattern()
      void updateShape(const L1RpcPatternsVec::const_iterator& pattern); 

      void setPatternsGroupType(RPCConst::TPatternType patternsGroupType);

      RPCConst::TPatternType getPatternsGroupType();

      void setGroupDescription(std::string groupDescription);

      std::string getGroupDescription() const;
      
  };



#endif
