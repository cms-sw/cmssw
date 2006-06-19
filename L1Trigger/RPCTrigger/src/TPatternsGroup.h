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
#include "L1Trigger/RPCTrigger/src/L1RpcPattern.h"
#include "L1Trigger/RPCTrigger/src/L1RpcLogCone.h"
  /** \class TPatternsGroup
 * Basic class for storing grouped patterns inside Pac.
 * In group (object of class TPatternsGroup) the patterns belonging to given
   * group are stored in PatternsVec. These patterns are use in trigger algorithm*/
  class TPatternsGroup {
    friend class L1RpcPac;
    protected:
      RPCParam::TPatternType PatternsGroupType;
    //L1RpcPatternsVec PatternsVec; //!< Vector of patterns.
      std::vector<L1RpcPatternsVec::const_iterator> PatternsItVec; //!< Vector of itereator on PatternsVec in Pac.
      L1RpcLogCone GroupShape; //!< Set LogStrips denotes strips beloging to the group.
      std::string GroupDescription;

    public:

      void AddPattern(const L1RpcPatternsVec::const_iterator& pattern);

    ///Updates GroupShape, i.e. sets to true strips belonging to the pattern. Coleed in AddPattern()
      void UpdateShape(const L1RpcPatternsVec::const_iterator& pattern); 

      void SetPatternsGroupType(RPCParam::TPatternType patternsGroupType);

      RPCParam::TPatternType GetPatternsGroupType();

      void SetGroupDescription(std::string groupDescription);

      std::string GetGroupDescription() const;
      
  };



#endif
