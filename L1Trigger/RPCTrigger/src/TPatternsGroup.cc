#include "L1Trigger/RPCTrigger/interface/TPatternsGroup.h"

//called by addPattern
void TPatternsGroup::updateShape(const RPCPattern::RPCPatVec::const_iterator& pattern) {
  for(int logPlane = RPCConst::m_FIRST_PLANE; logPlane <= RPCConst::m_LAST_PLANE; logPlane++) {
    if (pattern->getStripFrom(logPlane) != RPCConst::m_NOT_CONECTED) {
      int fromBit = pattern->getStripFrom(logPlane);
      int toBit = pattern->getStripTo(logPlane);
      for (int bitNumber = fromBit; bitNumber < toBit; bitNumber++)
        m_GroupShape.setLogStrip(logPlane, bitNumber);
    }
  }
}
/**
 *
 *The pattern is added to the m_PatternsVec, the m_GroupShape is updated (updateShape() is called).
 *
 */
void TPatternsGroup::addPattern(const RPCPattern::RPCPatVec::const_iterator& pattern){
  updateShape(pattern);
  m_PatternsItVec.push_back(pattern);
}

// Simple setters and getters
void TPatternsGroup::setPatternsGroupType(RPCPattern::TPatternType patternsGroupType){ 
  m_PatternsGroupType = patternsGroupType; 
}

void TPatternsGroup::setGroupDescription(std::string groupDescription){ 
  m_GroupDescription = groupDescription; 
}

std::string TPatternsGroup::getGroupDescription() const { 
  return m_GroupDescription; 
}

RPCPattern::TPatternType TPatternsGroup::getPatternsGroupType() const { 
  return m_PatternsGroupType; 
}
