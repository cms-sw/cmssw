/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/interface/RPCPattern.h"

RPCPattern::RPCPattern() {
  m_Number = -1; //empty pattern
  m_Code = 0;    
  m_Sign = 0;

  for(int logPlane = RPCConst::m_FIRST_PLANE; logPlane <= RPCConst::m_LAST_PLANE; logPlane++) {
    setStripFrom(logPlane, RPCConst::m_NOT_CONECTED);
    setStripTo(logPlane, RPCConst::m_NOT_CONECTED + 1);
  }
  //other parameters unset
}

void RPCPattern::setStripFrom(int logPlane, int stripFrom) { 
  m_Strips[logPlane].m_StripFrom = stripFrom; 
}

void RPCPattern::setStripTo(int logPlane, int stripTo) { m_Strips[logPlane].m_StripTo = stripTo; }

///First strip in range.
int RPCPattern::getStripFrom(int logPlane) const { //logic srtip
  return m_Strips[logPlane].m_StripFrom;
}

///Next-to-last strip in range.
int RPCPattern::getStripTo(int logPlane) const {  //logic srtip
  return m_Strips[logPlane].m_StripTo;
}

///Returns the stripFrom position w.r.t the first strip in ref plane.
int RPCPattern::getBendingStripFrom(int logPlane, int m_tower) const{
  if (m_Strips[logPlane].m_StripFrom == RPCConst::m_NOT_CONECTED){
    return  RPCConst::m_NOT_CONECTED;                                                   //expand
  }
  return m_Strips[logPlane].m_StripFrom 
      - m_Strips[RPCConst::m_REF_PLANE[m_tower]].m_StripFrom 
      - (RPCConst::m_LOGPLANE_SIZE[m_tower][logPlane] 
      - RPCConst::m_LOGPLANE_SIZE[m_tower][RPCConst::m_REF_PLANE[m_tower]])/2;
}

///Returns the stripTo position w.r.t the first strip in ref plane..
int RPCPattern::getBendingStripTo(int logPlane, int m_tower) const {
  if (m_Strips[logPlane].m_StripTo == RPCConst::m_NOT_CONECTED+1){
    return  RPCConst::m_NOT_CONECTED;                                                   //expand
  }
  
  return m_Strips[logPlane].m_StripTo 
               - m_Strips[RPCConst::m_REF_PLANE[m_tower]].m_StripFrom 
               - (RPCConst::m_LOGPLANE_SIZE[m_tower][logPlane] 
               - RPCConst::m_LOGPLANE_SIZE[m_tower][RPCConst::m_REF_PLANE[m_tower]])/2;
}

int RPCPattern::getCode() const{ return m_Code; }

int RPCPattern::getSign() const{ return m_Sign; }

int RPCPattern::getNumber() const{ return m_Number; }

RPCConst::TPatternType RPCPattern::getPatternType() const { return m_PatternType; }

int RPCPattern::getRefGroup() const { return m_RefGroup; }

int RPCPattern::getQualityTabNumber() const { return m_QualityTabNumber;}

void RPCPattern::setCode(int a) { m_Code = a;}

void RPCPattern::setSign(int a) { m_Sign = a;}

void RPCPattern::setNumber(int a) { m_Number = a;}

void RPCPattern::setPatternType(RPCConst::TPatternType patternType) { m_PatternType = patternType;}

void RPCPattern::setRefGroup(int refGroup) { m_RefGroup = refGroup; }

void RPCPattern::setQualityTabNumber(int qualityTabNumber) { m_QualityTabNumber = qualityTabNumber;}
