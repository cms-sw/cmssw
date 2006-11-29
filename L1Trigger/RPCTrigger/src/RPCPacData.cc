/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*  Porting to CMSSW - Tomasz Fruboes
*                                                                              *
*******************************************************************************/
#include <cmath>
#include <bitset>
#include "L1Trigger/RPCTrigger/src/RPCPacData.h"
#include "L1Trigger/RPCTrigger/src/RPCConst.h"

#include "L1Trigger/RPCTrigger/src/RPCException.h"

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
  
  std::string patFileName;
  
  RPCConst rpcconst;

//  if(patFilesDir.find("pat") != std::string::npos) {
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
//  }
/*
  else {
    throw RPCException("patFilesDir empty (no patterns)");
    //edm::LogError("RPCTrigger")<< "patFilesDir not containes XML";
  }
*/  

  m_TrackPatternsGroup.setGroupDescription("Track PatternsGroup");
    
}
/**
 *
 * @return the count af all patterns gropu, i.e. 1 + m_EnergeticPatternsGroupList.size(). 
 *
 */
int RPCPacData::getPatternsGroupCount () {
  return (1 + m_EnergeticPatternsGroupList.size() ); //1 = track pattrens group
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
  
  if (ret.empty()){
      throw RPCException("getEPatternsGroupShape(): groupNum to big!");
      //edm::LogError("RPCTrigger")<< "getEPatternsGroupShape(): groupNum to big!";
  }

  return ret;
}

void RPCPacData::insertQualityRecord(unsigned int qualityTabNumber,
                              unsigned short firedPlanes, short quality) {
  if(quality > m_MaxQuality)
    m_MaxQuality = quality;
  if(qualityTabNumber < m_QualityTabsVec.size() ) {
    m_QualityTabsVec[qualityTabNumber][firedPlanes] = quality;                   
  }
  else if(qualityTabNumber == m_QualityTabsVec.size() ) {
    // XXX - added cast (int)
    RPCConst::TQualityTab qualityTab((int)std::pow(2.0,RPCConst::m_LOGPLANES_COUNT), -1); //= new TQualityTab();
    m_QualityTabsVec.push_back(qualityTab);
    m_QualityTabsVec[qualityTabNumber][firedPlanes] = quality; 
  }
  else
    throw RPCException("insertQualityRecord(): wrong qualityTabNumber");
    //edm::LogError("RPCTrigger") << "insertQualityRecord(): wrong qualityTabNumber";
}


void RPCPacData::insertPatterns(const L1RpcPatternsVec& patternsVec) {
     
  RPCConst rpcconst;
  
  for(L1RpcPatternsVec::const_iterator patIt = patternsVec.begin(); patIt != patternsVec.end(); patIt++) { 
    if(patIt->getPatternType() == RPCConst::PAT_TYPE_T)
      m_TrackPatternsGroup.addPattern(patIt);
    else if (patIt->getPatternType() == RPCConst::PAT_TYPE_E) {
      TEPatternsGroupList::iterator iEGroup;
      for(iEGroup = m_EnergeticPatternsGroupList.begin();
          iEGroup != m_EnergeticPatternsGroupList.end(); iEGroup++)
        if(iEGroup->check(patIt) )
          break;
      if(iEGroup == m_EnergeticPatternsGroupList.end() ) {
        TEPatternsGroup newEGroup(patIt);
        newEGroup.setGroupDescription(
        //"EGroup #"+ rpcconst.intToString(m_EnergeticPatternsGroupList.size())+
        ", code: " + rpcconst.intToString(patIt->getCode()) +
        ", dir: " + rpcconst.intToString(patIt->getSign()) +
        ", refGroup: " + rpcconst.intToString(patIt->getRefGroup()) +
        ", qualityTabNumber: " + rpcconst.intToString(patIt->getQualityTabNumber()) );
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
    m_EnergeticPatternsGroupList.sort();           //to jest potrzebne, bo w run() przechodzi                                                                         //pierwszy paettern, ktory ma Maxymalna quality, wiec grupy musza byc 
    m_EnergeticPatternsGroupList.reverse();
  } 
}

void RPCPacData::init(const RPCPatternsParser& parser, const RPCConst::l1RpcConeCrdnts& coneCrdnts) {
  for(unsigned int i = 0; i < parser.getQualityVec().size(); i++) {    
    RPCPatternsParser::TQuality quality = parser.getQualityVec()[i];
    std::bitset<RPCConst::m_LOGPLANES_COUNT> qualBits(quality.m_FiredPlanes );
    unsigned short firedPlanes = qualBits.to_ulong();

    insertQualityRecord(quality.m_QualityTabNumber, firedPlanes, quality.m_QualityValue);  
  }

  m_PatternsVec = parser.getPatternsVec(coneCrdnts);
  insertPatterns(m_PatternsVec);
}

/*
RPCPacMuon RPCPacData::runTrackPatternsGroup(const RPCLogCone& cone) const {
  RPCPacMuon bestMuon;

  for(unsigned int vecNum = 0; vecNum < m_TrackPatternsGroup.m_PatternsItVec.size(); vecNum++) {
    unsigned short firedPlanes = 0;
    int firedPlanesCount = 0;
    unsigned short one = 1;
    const RPCPattern& pattern  = *(m_TrackPatternsGroup.m_PatternsItVec[vecNum]);
    for(int logPlane = RPCConst::m_FIRST_PLANE; logPlane < RPCConst::m_USED_PLANES_COUNT[m_ConeCrdnts.m_Tower]; logPlane++) {
      if (pattern.getStripFrom(logPlane) == RPCConst::m_NOT_CONECTED) {
        //firedPlanes[logPlane] = false; //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        continue;
      }
      int fromBit = pattern.getStripFrom(logPlane);
      int toBit = pattern.getStripTo(logPlane);
      for(int bitNumber = fromBit; bitNumber < toBit; bitNumber++) {
        if(cone.getLogStripState(logPlane, bitNumber) == true) {
          firedPlanes  = firedPlanes | one;
          firedPlanesCount++;
          break;
        }              
      }
 
      if( (RPCConst::m_USED_PLANES_COUNT[m_ConeCrdnts.m_Tower] - logPlane) == 3)
        if( firedPlanesCount == 0)
          break;
      
      one = one<<1;
    }

    if(firedPlanesCount >= 3) {
      short quality = m_QualityTabsVec[pattern.getQualityTabNumber()][firedPlanes];
      if(quality != -1) {
        if(quality >= bestMuon.getQuality() ) {
          RPCPacMuon bufMuon(pattern, quality, firedPlanes);
          if(bufMuon > bestMuon) {

            bestMuon = bufMuon;
            //if(bestMuon.getQuality() == m_MaxQuality ) //it can not be if there are patts of both sign sorted by sign
            //  return bestMuon;
          }
        }
      }
    }
  }
  return bestMuon;
}
*/

/*
RPCPacMuon RPCPacData::runEnergeticPatternsGroups(const RPCLogCone& cone) const {
  RPCPacMuon bestMuon;
  unsigned short firedPlanes = 0;
  int firedPlanesCount = 0;
  TEPatternsGroupList::const_iterator iEGroup = m_EnergeticPatternsGroupList.begin();
  for(; iEGroup != m_EnergeticPatternsGroupList.end(); iEGroup++) {
    firedPlanes = 0;
    firedPlanesCount = 0;
    unsigned short one = 1;
    for(int logPlane = RPCConst::m_FIRST_PLANE; logPlane < RPCConst::m_USED_PLANES_COUNT[m_ConeCrdnts.m_Tower]; logPlane++) {  //or po paskach ze stozka
      for(unsigned int bitNum = 0; bitNum < RPCConst::m_LOGPLANE_SIZE[abs(m_ConeCrdnts.m_Tower)][logPlane] ; bitNum++) {       
        if(iEGroup->m_GroupShape.getLogStripState(logPlane, bitNum) && cone.getLogStripState(logPlane, bitNum) ) {
          firedPlanes  = firedPlanes | one;
          firedPlanesCount++;
          break;     
        }
      }
      one = one << 1;
    }

    short quality = m_QualityTabsVec[iEGroup->m_QualityTabNumber][firedPlanes];
    if(quality == -1)
      continue;

    RPCPacMuon bufMuon;
    for(unsigned int vecNum = 0; vecNum < iEGroup->m_PatternsItVec.size(); vecNum++) {
      const L1RpcPatternsVec::const_iterator patternIt = iEGroup->m_PatternsItVec[vecNum];      
      const RPCPattern& pattern = *patternIt;     
      bool wasHit = false;
      unsigned short one1 = 1;
      for(int logPlane = RPCConst::m_FIRST_PLANE; logPlane < RPCConst::m_USED_PLANES_COUNT[m_ConeCrdnts.m_Tower]; logPlane++, one1 = one1<<1) {        
        if (pattern.getStripFrom(logPlane) == RPCConst::m_NOT_CONECTED) {
//          firedPlanes[logPlane] = false; //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
          continue;
        }
        if((firedPlanes & one1) != 0) {
          int fromBit = pattern.getStripFrom(logPlane);
          int toBit = pattern.getStripTo(logPlane);
          wasHit = false;
          for(int bitNumber = fromBit; bitNumber < toBit; bitNumber++)
            wasHit = wasHit || cone.getLogStripState(logPlane, bitNumber);
          if(!wasHit)
            break;
         }      
      }
      if(wasHit) {
        bufMuon.setAll(pattern, quality, firedPlanes);
        break;//if one pattern fits, thers no point to check other patterns from group
      }
    } //end of patterns loop
    if(bufMuon > bestMuon)
      bestMuon = bufMuon;
    if(bestMuon.getQuality() == m_MaxQuality)
      return bestMuon;
  }//end of EGroup loop
  return bestMuon;
}
*/
/** 
 *
 * Performs Pattern Comparator algorithm for hits from the cone.
 * Calls the runTrackPatternsGroup() and runEnergeticPatternsGroups().
 * @return found track candidate (empty if hits does not fit to eny pattern)
 *
*/
/*    
RPCPacMuon RPCPacData::run(const RPCLogCone& cone) const {  //symualcja
  //track
  RPCPacMuon bestMuon;
  if(m_TrackPatternsGroup.m_PatternsItVec.size() > 0)
    bestMuon = runTrackPatternsGroup(cone);

  //energetic
  if(m_EnergeticPatternsGroupList.size() > 0) {
    RPCPacMuon bufMuon = runEnergeticPatternsGroups(cone);
    if(bufMuon > bestMuon)
      bestMuon = bufMuon;
  }

  bestMuon.setConeCrdnts(m_CurrConeCrdnts);
  bestMuon.setLogConeIdx(cone.getIdx());
  int refStripNum = getPattern(bestMuon.getPatternNum()).getStripFrom(RPCConst::m_REF_PLANE[abs(m_CurrConeCrdnts.m_Tower)]) + m_CurrConeCrdnts.m_LogSector * 96 + m_CurrConeCrdnts.m_LogSegment * 8;
	bestMuon.setRefStripNum(refStripNum);
  return bestMuon;
}
*/
//------------------------------------------------------------------------------



