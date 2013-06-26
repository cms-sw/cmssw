/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/interface/RPCLogCone.h"
#include "L1Trigger/RPCTrigger/interface/RPCException.h"

#include <iostream>
#include <iomanip>
#include <sstream>

/**
 *
 * Default constructor. No hits, no muon.
 *
*/
RPCLogCone::RPCLogCone():
    m_ConeCrdnts()
{
  m_LogPlanesVec.assign(RPCConst::m_LOGPLANES_COUNT, TLogPlane());
  m_MuonCode = 0;
  m_MuonSign = 0;
}
/**
 *
 * Constructor. Cone coordinates are set.
 *
*/

RPCLogCone::RPCLogCone(int tower, int logSector, int logSegment):
    m_ConeCrdnts(tower, logSector, logSegment)
{
  m_LogPlanesVec.assign(RPCConst::m_LOGPLANES_COUNT, TLogPlane());
  m_MuonCode = 0;
  m_MuonSign = 0;
}
/**
 *
 * Copying Constructor
 *
*/
RPCLogCone::RPCLogCone(const RPCLogHit &logHit)
{
  m_LogPlanesVec.assign(RPCConst::m_LOGPLANES_COUNT, TLogPlane());

  m_ConeCrdnts = logHit.getConeCrdnts();

  m_MuonCode = 0;
  m_MuonSign = 0;

  setLogStrip(logHit.getlogPlaneNumber() -1, logHit.getStripNumberInCone(), logHit.getDigiIdx());
}

RPCLogCone::RPCLogCone(const unsigned long long &pat, int tower, int logSector, int logSegment):
      m_ConeCrdnts(tower, logSector, logSegment)
{
  m_LogPlanesVec.assign(RPCConst::m_LOGPLANES_COUNT, TLogPlane());
  m_MuonCode = 0;
  m_MuonSign = 0;

  unsigned long long int mask = 255; // (first 8 bits)
  int shift = 0;

  //std::cout << "Decompressing pattern: " << pat << std::endl;
  for (int logplane = RPCConst::m_FIRST_PLANE;
           logplane != RPCConst::m_USED_PLANES_COUNT[std::abs(getTower())];
         ++logplane  )
  {
      unsigned int strip = (pat & (mask<<shift) ) >> shift;
      //std::cout << logplane << " " << strip << std::endl;
      shift += 8;
      // We should prob. use m_NOT_CONNECTED value
      if (strip != RPCConst::m_LOGPLANE_SIZE[std::abs(getTower())][logplane])
        setLogStrip(logplane,strip);
  }
}

unsigned long long RPCLogCone::getCompressedCone(){

    unsigned long long int pattern = 0;
    int shift = 0;

    for (int logplane = RPCConst::m_FIRST_PLANE;
             logplane != RPCConst::m_USED_PLANES_COUNT[std::abs(getTower())];
             logplane++  )
     {
       unsigned long long int strip;
       if (getHitsCnt(logplane)==0) {
          // We need to mark somehow, that plane is empty (strip 0 is not fired)
          strip = RPCConst::m_LOGPLANE_SIZE[std::abs(getTower())][logplane];
       }
       else if (getHitsCnt(logplane)==1) {
          RPCLogCone::TLogPlane lp = getLogPlane(logplane);
          strip = lp.begin()->first;
       }
       else {
          throw RPCException("To many hits in logcone");
       }
          pattern = pattern | (strip << shift);
          shift += 8;
     }

   //std::cout << " Compressed cone: "   << pattern << std::endl;
   return pattern;
}


std::string RPCLogCone::toString() const {
  std::ostringstream ostr;
  ostr << "\n       ======================> TOWER = ";
  ostr<<std::setw(2)
      <<m_ConeCrdnts.m_Tower
      <<", m_LogSector = "
      <<m_ConeCrdnts.m_LogSector
      <<",  m_LogSegment = "
      <<m_ConeCrdnts.m_LogSegment
      <<" <======================="<< std::endl;

  std::string spacer;

  for (int logPlane = RPCConst::m_LAST_PLANE; logPlane >= RPCConst::m_FIRST_PLANE; logPlane--) {
    ostr<<RPCConst::m_LOGPLANE_STR[logPlane]<<" ";
    spacer.assign((72 - RPCConst::m_LOGPLANE_SIZE[abs(m_ConeCrdnts.m_Tower)][logPlane])/2, ' ');
    ostr<<spacer;
  
    for(int i = RPCConst::m_LOGPLANE_SIZE[abs(m_ConeCrdnts.m_Tower)][logPlane]-1; i >=0; i--) {
      if(getLogStripState(logPlane, i))
         ostr<<"X";
      else {
        if(i%8 == 0)
          ostr<<i%10;
        else
    ostr<<"."; 
      }  
    }  

    ostr<<std::endl;
  }
 
  ostr<< std::endl;
  return ostr.str();
}
void RPCLogCone::shift(int pos) {
  int shiftPos;
  for(int logPlane = RPCConst::m_FIRST_PLANE; logPlane <= RPCConst::m_LAST_PLANE; logPlane++) {
    TLogPlane shifted;
    for(TLogPlane::iterator it = m_LogPlanesVec[logPlane].begin();
        it != m_LogPlanesVec[logPlane].end();
        it++)
    {
      shiftPos = it->first + pos;
 /*     std::cout << shiftPos << " "
                << RPCConst::m_LOGPLANE_SIZE[abs(m_ConeCrdnts.m_Tower)] 
                << std::endl;*/
      if ( shiftPos >= 0 && shiftPos < (int)RPCConst::m_LOGPLANE_SIZE[abs(m_ConeCrdnts.m_Tower)][logPlane])
        shifted.insert(TLogPlane::value_type(shiftPos, it->second));
    }
    m_LogPlanesVec[logPlane] = shifted;
  }
}
// 
//#############################################################################################
//
//  Simple getters and setters
//
//#############################################################################################
RPCLogCone::TLogPlane RPCLogCone::getLogPlane(int logPlane) const { 
  return m_LogPlanesVec[logPlane]; 
}

///Gets fired strips count in given logPlane.
int RPCLogCone::getHitsCnt(int logPlane) const {
  return m_LogPlanesVec[logPlane].size();
}

/// sets pt code of muon that fired the strips */
void RPCLogCone::setMuonCode(int code) { m_MuonCode = code; }

/** @return pt code of muon that fired the strips */
int RPCLogCone::getMuonCode() const { return m_MuonCode; }

void RPCLogCone::setMuonSign(int sign) { m_MuonSign = sign; }

int RPCLogCone::getMuonSign() const { return m_MuonSign; }

int RPCLogCone::getTower() const { return m_ConeCrdnts.m_Tower; }

int RPCLogCone::getLogSector() const { return m_ConeCrdnts.m_LogSector; }

int RPCLogCone::getLogSegment() const { return m_ConeCrdnts.m_LogSegment; }

RPCConst::l1RpcConeCrdnts RPCLogCone::getConeCrdnts() const { return m_ConeCrdnts; }

void RPCLogCone::setIdx(int index) { m_Index = index; }

int RPCLogCone::getIdx() const { return m_Index; }
  
void RPCLogCone::setLogStrip(int logPlane, int logStripNum, int m_digiIdx) {
//m_LogPlanesVec[logPlane].insert(logStripNum);
//m_LogPlanesVec[logPlane].insert(TLogPlane::value_type(logStripNum, vector<int>()));
  m_LogPlanesVec[logPlane][logStripNum].push_back(m_digiIdx);
}

void RPCLogCone::setLogStrip(int logPlane, int logStripNum) {
  m_LogPlanesVec[logPlane].insert(TLogPlane::value_type(logStripNum, std::vector<int>()));
}


bool RPCLogCone::getLogStripState(int logPlane, unsigned int logStripNum)  const {
  return m_LogPlanesVec[logPlane].count(logStripNum);
}

  
/**
 *
 * Adds a loghit to a cone
 * 
*/
bool RPCLogCone::addLogHit(const RPCLogHit &logHit) {
  
  if (m_ConeCrdnts.m_Tower == logHit.getTower() &&
      m_ConeCrdnts.m_LogSector == logHit.getLogSector() &&
      m_ConeCrdnts.m_LogSegment == logHit.getLogSegment()) 
  {
    setLogStrip(logHit.getlogPlaneNumber()-1, logHit.getStripNumberInCone(), logHit.getDigiIdx());
    return true;
  }
  else
    return false;
}


std::vector<int> RPCLogCone::getLogStripDigisIdxs(int logPlane, unsigned int logStripNum) const {
  TLogPlane::const_iterator it = m_LogPlanesVec[logPlane].find(logStripNum); 
  if(it != m_LogPlanesVec[logPlane].end())
    return it->second;
  else
    return std::vector<int>();
}


bool RPCLogCone::isPlaneFired(int logPlane) const {
  if(m_LogPlanesVec[logPlane].size() == 0)
    return false; 
  else
    return true;  
}

int RPCLogCone::getFiredPlanesCnt() const{
  int firedPlanes = 0;
  for(int logPlane = RPCConst::m_FIRST_PLANE;
      logPlane < RPCConst::m_USED_PLANES_COUNT[abs(m_ConeCrdnts.m_Tower)];
      logPlane++)
  {
    firedPlanes = firedPlanes + isPlaneFired(logPlane);
  }
  return firedPlanes;
}


int RPCLogCone::possibleTrigger() const {
  int triggerType = 0; //0 - trigger not possible
                        //1 - 3/4 (3 inner planes fired)
                        //2 - 4/6 (four palnes fired)
  int firedPlanes = 0;

  int logPlane = RPCConst::m_FIRST_PLANE;
  for( ; logPlane <= RPCConst::m_LOGPLANE4; logPlane++) {
    firedPlanes = firedPlanes + isPlaneFired(logPlane);
  }
  if(firedPlanes >= 3)
    triggerType = 1;

  for( ;
      logPlane < RPCConst::m_USED_PLANES_COUNT[abs(m_ConeCrdnts.m_Tower)];
      logPlane++)
  {
    firedPlanes = firedPlanes + isPlaneFired(logPlane);
  }
  if(firedPlanes >= 4)
    triggerType = 2;

  return triggerType;
}
