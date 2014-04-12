/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*  Porting to CMSSW - Tomasz Fruboes
*                                                                              *
*******************************************************************************/
#include <cmath>
#include <bitset>
#include "L1Trigger/RPCTrigger/interface/RPCPacData.h"
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

#include "L1Trigger/RPCTrigger/interface/RPCException.h"

#include <iostream>

//#include "FWCore/MessageLogger/interface/MessageLogger.h"


/** 
 *
 * Constructor required by RPCPacManager.
 * @param patFilesDir -  the directory conataing m_PAC definition file.
 * It should containe file for this m_PAC, defined by tower, logSector, logSegment,
 * named pacPat_t<tower>sc<logSector>sg<logSegment>.vhd
 * Containers: m_EnergeticPatternsGroupList and m_TrackPatternsGroup are
 * filled with patterns from file (the method ParsePatternFile() is called).
 * 
 */
RPCPacData::RPCPacData(std::string patFilesDir, int tower, int logSector, int logSegment){
  
 //: RPCPacBase(tower, logSector, logSegment) { 
  m_MaxQuality=0;
  std::string patFileName;
  
  RPCConst rpcconst;


  patFileName = patFilesDir 
      + "pacPat_t" + rpcconst.intToString(tower) 
      + "sc" + rpcconst.intToString(logSector) 
      + "sg" + rpcconst.intToString(logSegment) 
      + ".xml";

  RPCConst::l1RpcConeCrdnts coneCrds;
  coneCrds.m_Tower=tower;
  coneCrds.m_LogSector=logSector;
  coneCrds.m_LogSegment=logSegment;
        
  RPCPatternsParser parser;
  parser.parse(patFileName);
  init(parser, coneCrds);

  m_TrackPatternsGroup.setGroupDescription("Track PatternsGroup");
    
}
/**
 *
 * Construct from given qualities and patterns
 *
 */
RPCPacData::RPCPacData(const RPCPattern::RPCPatVec &patVec, 
                       const RPCPattern::TQualityVec &qualVec) :
  m_MaxQuality(0)
{
  for(unsigned int i = 0; i < qualVec.size(); ++i) {    
    RPCPattern::TQuality quality = qualVec[i];
    std::bitset<RPCConst::m_LOGPLANES_COUNT> qualBits(quality.m_FiredPlanes);
    unsigned short firedPlanes = qualBits.to_ulong();

    insertQualityRecord(quality.m_QualityTabNumber, firedPlanes, quality.m_QualityValue);  
  }

  
  insertPatterns(patVec);

   
}



RPCPacData::RPCPacData(const L1RPCConfig * rpcconf, const int tower, const int sector, const int segment):
  m_MaxQuality(0)
{

    for (unsigned int iqual=0; iqual<rpcconf->m_quals.size(); iqual++){

      if (rpcconf->m_quals[iqual].m_tower != tower ||  
          rpcconf->m_quals[iqual].m_logsector != sector ||
          rpcconf->m_quals[iqual].m_logsegment != segment )  continue;

      RPCPattern::TQuality quality =rpcconf->m_quals[iqual];
      std::bitset<RPCConst::m_LOGPLANES_COUNT> qualBits(quality.m_FiredPlanes);
      unsigned short firedPlanes = qualBits.to_ulong();
      insertQualityRecord(quality.m_QualityTabNumber, firedPlanes, quality.m_QualityValue);

    }

  insertPatterns(rpcconf->m_pats,tower,sector,segment);

}





/**
 *
 * @return the count af all patterns gropu, i.e. 1 + m_EnergeticPatternsGroupList.size(). 
 *
 */
int RPCPacData::getPatternsGroupCount() {
  return(1 + m_EnergeticPatternsGroupList.size()); //1 = track pattrens group
}
/**
 *
 * @return pattern stored in m_PatternsVec.
 * Needed for patterns explorer.
 *
 */
RPCPattern RPCPacData::getPattern(int patNum) const {
  if(m_PatternsVec.size() == 0) {
  
    throw RPCException("GetPattren(): Patterns vec is empty, mayby it was not filled!");
    //edm::LogError("RPCTrigger") << "GetPattren(): Patterns vec is empty, mayby it was not filled!";
    
  }
  return m_PatternsVec[patNum];
  
}
/**
 * 
 *@return the count of patterns stored in m_PatternsVec.
 *
 */
int RPCPacData::getPatternsCount() {
  return m_PatternsVec.size();
}
/**
 *
 *@return true, if logStrip defined by logStripNum and logPlane  belongs to the
 * m_TrackPatternsGroup. 
 *
*/
bool RPCPacData::getTPatternsGroupShape(int logPlane, int logStripNum) {
  return m_TrackPatternsGroup.m_GroupShape.getLogStripState(logPlane, logStripNum);
}
/** 
 *
 * @return true, if logStrip defined by logStripNum and logPlane  belongs to the
 * EPatternsGroup from m_EnergeticPatternsGroupList defined by groupNum. 
 * 
*/
bool RPCPacData::getEPatternsGroupShape(int groupNum, int logPlane, int bitNum) {
  TEPatternsGroupList::const_iterator iEGroup = m_EnergeticPatternsGroupList.begin();
  int i = 0;
  for(; iEGroup != m_EnergeticPatternsGroupList.end(); iEGroup++, i++) {
    if(i == groupNum)
      return iEGroup->m_GroupShape.getLogStripState(logPlane, bitNum);
  }
  throw RPCException("getEPatternsGroupShape(): groupNum to big!");
  //edm::LogError("RPCTrigger")<< "getEPatternsGroupShape(): groupNum to big!";
  return false; // XXX - TMF
}

std::string RPCPacData::getPatternsGroupDescription(int patternGroupNum) {
  
  RPCConst rpcconst;
  
  std::string ret;
  if(patternGroupNum == -1)
    ret =  m_TrackPatternsGroup.getGroupDescription();
  else  {
    TEPatternsGroupList::iterator iEGroup = m_EnergeticPatternsGroupList.begin();
    int i = 0;
    for(; iEGroup != m_EnergeticPatternsGroupList.end(); iEGroup++, i++) {
      if(i == patternGroupNum)
        ret = "EGroup #"+ rpcconst.intToString(i)+iEGroup->getGroupDescription();
    }
    
  }
  
  if(ret.empty()){
      throw RPCException("getEPatternsGroupShape(): groupNum to big!");
      //edm::LogError("RPCTrigger")<< "getEPatternsGroupShape(): groupNum to big!";
  }

  return ret;
}

void RPCPacData::insertQualityRecord(unsigned int qualityTabNumber,
                              unsigned short firedPlanes, short quality) {
  
  if(quality > m_MaxQuality)
    m_MaxQuality = quality;
  if(qualityTabNumber < m_QualityTabsVec.size()) {
    m_QualityTabsVec[qualityTabNumber][firedPlanes] = quality;                   
  }
  else if(qualityTabNumber == m_QualityTabsVec.size()) {
    // XXX - added cast(int)

    //= new TQualityTab();
    RPCConst::TQualityTab qualityTab((int)std::pow(2.0,RPCConst::m_LOGPLANES_COUNT), -1); 
    m_QualityTabsVec.push_back(qualityTab);
    m_QualityTabsVec[qualityTabNumber][firedPlanes] = quality; 
  }
  else
    throw RPCException("insertQualityRecord(): wrong qualityTabNumber");
    //edm::LogError("RPCTrigger") << "insertQualityRecord(): wrong qualityTabNumber";
}


void RPCPacData::insertPatterns(const RPCPattern::RPCPatVec& patternsVec, const int tower, const int sector, const int segment) {
   
  bool ignorePos = false;
  if ( tower == 99 || sector == 99 || segment == 99) ignorePos = true; 
  
  RPCConst rpcconst;
  
  for(RPCPattern::RPCPatVec::const_iterator patIt = patternsVec.begin();
      patIt != patternsVec.end();
      patIt++)
  {
    if (!ignorePos &&
         (patIt->getTower() != tower  
          || patIt->getLogSector() != sector  
          || patIt->getLogSegment() != segment) ) continue;
    
    if(patIt->getPatternType() == RPCPattern::PAT_TYPE_T)
      m_TrackPatternsGroup.addPattern(patIt);
    else if(patIt->getPatternType() == RPCPattern::PAT_TYPE_E) {
      TEPatternsGroupList::iterator iEGroup;
      for(iEGroup = m_EnergeticPatternsGroupList.begin();
          iEGroup != m_EnergeticPatternsGroupList.end(); iEGroup++)
        if(iEGroup->check(patIt))
          break;
      if(iEGroup == m_EnergeticPatternsGroupList.end()) {
        TEPatternsGroup newEGroup(patIt);
        newEGroup.setGroupDescription(
        //"EGroup #"+ rpcconst.intToString(m_EnergeticPatternsGroupList.size())+
        ", code: " + rpcconst.intToString(patIt->getCode()) +
        ", dir: " + rpcconst.intToString(patIt->getSign()) +
        ", refGroup: " + rpcconst.intToString(patIt->getRefGroup()) +
        ", qualityTabNumber: " + rpcconst.intToString(patIt->getQualityTabNumber()));
        m_EnergeticPatternsGroupList.push_back(newEGroup);
      }
      else
       iEGroup->addPattern(patIt);
    }
    else
      throw RPCException("InsertPattern(): unsupported pattern type");
      //edm::LogError("RPCTrigger") << "InsertPattern(): unsupported pattern type";
  }  

  if(m_EnergeticPatternsGroupList.size() != 0) {
     
    m_EnergeticPatternsGroupList.sort();  //to jest potrzebne, bo w run() przechodzi
                                          //pierwszy paettern, ktory ma Maxymalna quality, wiec
                                          //grupy musza byc
    m_EnergeticPatternsGroupList.reverse();
  } 
}

void RPCPacData::init(const RPCPatternsParser& parser, const RPCConst::l1RpcConeCrdnts& coneCrdnts) {
  for(unsigned int i = 0; i < parser.getQualityVec().size(); i++) {    
    RPCPattern::TQuality quality = parser.getQualityVec()[i];
    std::bitset<RPCConst::m_LOGPLANES_COUNT> qualBits(quality.m_FiredPlanes);
    unsigned short firedPlanes = qualBits.to_ulong();

    insertQualityRecord(quality.m_QualityTabNumber, firedPlanes, quality.m_QualityValue);  
  }

  m_PatternsVec = parser.getPatternsVec(coneCrdnts);
  insertPatterns(m_PatternsVec);
}
