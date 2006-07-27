/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/L1RpcPattern.h"

L1RpcPattern::L1RpcPattern() {
  Number = -1; //empty pattern
  Code = 0;    
  Sign = 0;

  for(int logPlane = rpcparam::FIRST_PLANE; logPlane <= rpcparam::LAST_PLANE; logPlane++) {
    SetStripFrom(logPlane, rpcparam::NOT_CONECTED);
    SetStripTo(logPlane, rpcparam::NOT_CONECTED + 1);
  }
  //other parameters unset
}

void L1RpcPattern::SetStripFrom(int logPlane, int stripFrom) { Strips[logPlane].StripFrom = stripFrom; }

void L1RpcPattern::SetStripTo(int logPlane, int stripTo) { Strips[logPlane].StripTo = stripTo; }

///First strip in range.
int L1RpcPattern::GetStripFrom(int logPlane) const { //logic srtip
  return Strips[logPlane].StripFrom;
}

///Next-to-last strip in range.
int L1RpcPattern::GetStripTo(int logPlane) const {  //logic srtip
  return Strips[logPlane].StripTo;
}

///Returns the stripFrom position w.r.t the first strip in ref plane.
int L1RpcPattern::GetBendingStripFrom(int logPlane, int tower) {
  if (Strips[logPlane].StripFrom == rpcparam::NOT_CONECTED){
    return  rpcparam::NOT_CONECTED;                                                   //expand
  }
  return Strips[logPlane].StripFrom 
      - Strips[rpcparam::REF_PLANE[tower]].StripFrom 
      - (rpcparam::LOGPLANE_SIZE[tower][logPlane] 
      - rpcparam::LOGPLANE_SIZE[tower][rpcparam::REF_PLANE[tower]])/2;
}

///Returns the stripTo position w.r.t the first strip in ref plane..
int L1RpcPattern::GetBendingStripTo(int logPlane, int tower) {
  if (Strips[logPlane].StripTo == rpcparam::NOT_CONECTED+1)
    return  rpcparam::NOT_CONECTED;                                                   //expand
  return Strips[logPlane].StripTo - Strips[rpcparam::REF_PLANE[tower]].StripFrom - (rpcparam::LOGPLANE_SIZE[tower][logPlane] - rpcparam::LOGPLANE_SIZE[tower][rpcparam::REF_PLANE[tower]])/2;
}

int L1RpcPattern::GetCode() const{ return Code; }

int L1RpcPattern::GetSign() const{ return Sign; }

int L1RpcPattern::GetNumber() const{ return Number; }

rpcparam::TPatternType L1RpcPattern::GetPatternType() const { return PatternType; }

int L1RpcPattern::GetRefGroup() const { return RefGroup; }

int L1RpcPattern::GetQualityTabNumber() const { return QualityTabNumber;}

void L1RpcPattern::SetCode(int a) { Code = a;}

void L1RpcPattern::SetSign(int a) { Sign = a;}

void L1RpcPattern::SetNumber(int a) { Number = a;}

void L1RpcPattern::SetPatternType(rpcparam::TPatternType patternType) { PatternType = patternType; }

void L1RpcPattern::SetRefGroup(int refGroup) { RefGroup = refGroup; }

void L1RpcPattern::SetQualityTabNumber(int qualityTabNumber ) { QualityTabNumber = qualityTabNumber;}
