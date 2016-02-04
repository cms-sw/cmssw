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
#include "CondFormats/L1TObjects/interface/RPCPattern.h"
#include "L1Trigger/RPCTrigger/interface/RPCLogCone.h"
  /** \class TPatternsGroup
 * Basic class for storing grouped patterns inside Pac.
 * In group (object of class TPatternsGroup) the patterns belonging to given
   * group are stored in m_PatternsVec. These patterns are use in trigger algorithm*/
  class TPatternsGroup {
    friend class RPCPacData;
    friend class RPCPac;
    
    public:
      void addPattern(const RPCPattern::RPCPatVec::const_iterator& pattern);

    ///Updates m_GroupShape, i.e. sets to true strips belonging to the pattern. Coleed in addPattern()
      void updateShape(const RPCPattern::RPCPatVec::const_iterator& pattern); 

      void setPatternsGroupType(RPCPattern::TPatternType patternsGroupType);

      RPCPattern::TPatternType getPatternsGroupType() const;

      void setGroupDescription(std::string groupDescription);

      std::string getGroupDescription() const;
          
    protected:
      RPCPattern::TPatternType m_PatternsGroupType;
    //L1RpcPatternsVec m_PatternsVec; //!< Vector of patterns.
      
      //!< Vector of itereator on m_PatternsVec in Pac.
      std::vector<RPCPattern::RPCPatVec::const_iterator> m_PatternsItVec; 
      
      //!< Set LogStrips denotes strips beloging to the group.
      RPCLogCone m_GroupShape; 
      
      std::string m_GroupDescription;

      
  };
#endif
