#include "L1Trigger/RPCTrigger/src/TPatternsGroup.h"

void TPatternsGroup::UpdateShape(const L1RpcPatternsVec::const_iterator& pattern) { //colled by AddPattern
  for(int logPlane = L1RpcConst::FIRST_PLANE; logPlane <= L1RpcConst::LAST_PLANE; logPlane++) {
    if (pattern->GetStripFrom(logPlane) != L1RpcConst::NOT_CONECTED) {
      int fromBit = pattern->GetStripFrom(logPlane);
      int toBit = pattern->GetStripTo(logPlane);
      for (int bitNumber = fromBit; bitNumber < toBit; bitNumber++)
        GroupShape.SetLogStrip(logPlane, bitNumber);
    }
  }
}
/**
 *
 *The pattern is added to the PatternsVec, the GroupShape is updated (UpdateShape() is called).
 *
 */
void TPatternsGroup::AddPattern(const L1RpcPatternsVec::const_iterator& pattern) {
  UpdateShape(pattern);
  PatternsItVec.push_back(pattern);
}

// Simple setters and getters
void TPatternsGroup::SetPatternsGroupType(L1RpcConst::TPatternType patternsGroupType) { PatternsGroupType = patternsGroupType; }

void TPatternsGroup::SetGroupDescription(std::string groupDescription) { GroupDescription = groupDescription; }

std::string TPatternsGroup::GetGroupDescription() const { return GroupDescription; }

L1RpcConst::TPatternType TPatternsGroup::GetPatternsGroupType() { return PatternsGroupType; }
