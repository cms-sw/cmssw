#include "L1Trigger/RPCTrigger/src/TEPatternsGroup.h"

/**
 *
 * Creates new patterns group. The pattern is added to the group and defined
 * its Code, Sign, RefGroup, QualityTabNumber. 
 *
 */
 
TEPatternsGroup::TEPatternsGroup(const L1RpcPatternsVec::const_iterator& pattern) {
  AddPattern(pattern);
  PatternsGroupType = rpcparam::PAT_TYPE_E;
  QualityTabNumber = pattern->GetQualityTabNumber(); //it is uded in PAC algorithm, so we want to have fast acces.
}


bool TEPatternsGroup::Check(const L1RpcPatternsVec::const_iterator& pattern) {
  if(PatternsItVec[0]->GetRefGroup() == pattern->GetRefGroup() &&
     PatternsItVec[0]->GetCode() == pattern->GetCode() &&
     PatternsItVec[0]->GetSign() == pattern->GetSign() &&
     PatternsItVec[0]->GetQualityTabNumber() == pattern->GetQualityTabNumber() )
    return true;
  return false;
}


bool TEPatternsGroup::operator < (const TEPatternsGroup& ePatternsGroup) const {
  if( this->PatternsItVec[0]->GetCode() < ePatternsGroup.PatternsItVec[0]->GetCode() )
    return true;
  else if( this->PatternsItVec[0]->GetCode() > ePatternsGroup.PatternsItVec[0]->GetCode() )
    return false;
  else { //==
    if(this->PatternsItVec[0]->GetQualityTabNumber() > ePatternsGroup.PatternsItVec[0]->GetQualityTabNumber())
      return true;
    else if(this->PatternsItVec[0]->GetQualityTabNumber() < ePatternsGroup.PatternsItVec[0]->GetQualityTabNumber())
      return false;
    else { //==
      if( this->PatternsItVec[0]->GetSign() < ePatternsGroup.PatternsItVec[0]->GetSign() )
        return true;
      else if( this->PatternsItVec[0]->GetSign() > ePatternsGroup.PatternsItVec[0]->GetSign() )
        return false;
      else { //==
        if(this->PatternsItVec[0]->GetRefGroup() < ePatternsGroup.PatternsItVec[0]->GetRefGroup())
          return true;
        else //if(this->RefGroup < ePatternsGroup.RefGroup)
          return false;
      }
    }
  }
}
