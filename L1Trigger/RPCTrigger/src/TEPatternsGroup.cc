#include "L1Trigger/RPCTrigger/interface/TEPatternsGroup.h"

/**
 *
 * Creates new patterns group. The pattern is added to the group and defined
 * its m_Code, m_Sign, m_RefGroup, m_QualityTabNumber. 
 *
 */
 
TEPatternsGroup::TEPatternsGroup(const RPCPattern::RPCPatVec::const_iterator& pattern) {
  addPattern(pattern);
  m_PatternsGroupType = RPCPattern::PAT_TYPE_E;
  m_QualityTabNumber = pattern->getQualityTabNumber(); //it is uded in m_PAC algorithm, so we want to have fast acces.
}


bool TEPatternsGroup::check(const RPCPattern::RPCPatVec::const_iterator& pattern) {
  if(m_PatternsItVec[0]->getRefGroup() == pattern->getRefGroup() &&
     m_PatternsItVec[0]->getCode() == pattern->getCode() &&
     m_PatternsItVec[0]->getSign() == pattern->getSign() &&
     m_PatternsItVec[0]->getQualityTabNumber() == pattern->getQualityTabNumber() )
    return true;
  return false;
}


bool TEPatternsGroup::operator < (const TEPatternsGroup& ePatternsGroup) const {
  if( this->m_PatternsItVec[0]->getCode() < ePatternsGroup.m_PatternsItVec[0]->getCode() )
    return true;
  else if( this->m_PatternsItVec[0]->getCode() > ePatternsGroup.m_PatternsItVec[0]->getCode() )
    return false;
  else { //==
    if(this->m_PatternsItVec[0]->getQualityTabNumber() > ePatternsGroup.m_PatternsItVec[0]->getQualityTabNumber())
      return true;
    else if(this->m_PatternsItVec[0]->getQualityTabNumber() < ePatternsGroup.m_PatternsItVec[0]->getQualityTabNumber())
      return false;
    else { //==
      if( this->m_PatternsItVec[0]->getSign() < ePatternsGroup.m_PatternsItVec[0]->getSign() )
        return true;
      else if( this->m_PatternsItVec[0]->getSign() > ePatternsGroup.m_PatternsItVec[0]->getSign() )
        return false;
      else { //==
        if(this->m_PatternsItVec[0]->getRefGroup() < ePatternsGroup.m_PatternsItVec[0]->getRefGroup())
          return true;
        else //if(this->m_RefGroup < ePatternsGroup.m_RefGroup)
          return false;
      }
    }
  }
}
