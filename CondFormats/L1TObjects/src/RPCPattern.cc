/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "CondFormats/L1TObjects/interface/RPCPattern.h"

RPCPattern::RPCPattern() {
  m_Number = -1;  //empty pattern
  m_Tower = 0;
  m_LogSector = 0;
  m_LogSegment = 0;
  m_Code = 0;
  m_Sign = 0;
  m_RefGroup = 0;
  m_QualityTabNumber = 0;
  m_PatternType = 0;

  for (int logPlane = m_FIRST_PLANE; logPlane <= m_LAST_PLANE; logPlane++) {
    setStripFrom(logPlane, m_NOT_CONECTED);
    setStripTo(logPlane, m_NOT_CONECTED + 1);
  }
  //other parameters unset
}

RPCPattern::RPCPattern(int tower, int sector, int segment) {
  m_Number = -1;  //empty pattern
  m_Tower = tower;
  m_LogSector = sector;
  m_LogSegment = segment;
  m_Code = 0;
  m_Sign = 0;
  m_RefGroup = 0;
  m_QualityTabNumber = 0;
  m_PatternType = 0;

  for (int logPlane = m_FIRST_PLANE; logPlane <= m_LAST_PLANE; logPlane++) {
    setStripFrom(logPlane, m_NOT_CONECTED);
    setStripTo(logPlane, m_NOT_CONECTED + 1);
  }
  //other parameters unset
}

void RPCPattern::setStripFrom(int logPlane, int stripFrom) { m_Strips[logPlane].m_StripFrom = stripFrom; }

void RPCPattern::setStripTo(int logPlane, int stripTo) { m_Strips[logPlane].m_StripTo = stripTo; }

///First strip in range.
int RPCPattern::getStripFrom(int logPlane) const {  //logic srtip
  return m_Strips[logPlane].m_StripFrom;
}

///Next-to-last strip in range.
int RPCPattern::getStripTo(int logPlane) const {  //logic srtip
  return m_Strips[logPlane].m_StripTo;
}

int RPCPattern::getTower() const { return m_Tower; }
int RPCPattern::getLogSector() const { return m_LogSector; }
int RPCPattern::getLogSegment() const { return m_LogSegment; }

int RPCPattern::getCode() const { return m_Code; }

int RPCPattern::getSign() const { return m_Sign; }

int RPCPattern::getNumber() const { return m_Number; }

RPCPattern::TPatternType RPCPattern::getPatternType() const { return m_PatternType; }

int RPCPattern::getRefGroup() const { return m_RefGroup; }

int RPCPattern::getQualityTabNumber() const { return m_QualityTabNumber; }

void RPCPattern::setCoords(int tower, int sector, int segment) {
  m_Tower = tower;
  m_LogSector = sector;
  m_LogSegment = segment;
}

void RPCPattern::setCode(int a) { m_Code = a; }

void RPCPattern::setSign(int a) { m_Sign = a; }

void RPCPattern::setNumber(int a) { m_Number = a; }

void RPCPattern::setPatternType(TPatternType patternType) { m_PatternType = patternType; }

void RPCPattern::setRefGroup(int refGroup) { m_RefGroup = refGroup; }

void RPCPattern::setQualityTabNumber(int qualityTabNumber) { m_QualityTabNumber = qualityTabNumber; }
