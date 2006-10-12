/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*  Porting to CMSSW - Tomasz Fruboes
*                                                                              *
*******************************************************************************/
#include <math.h>
#include <bitset>
#include "L1Trigger/RPCTrigger/src/L1RpcPac.h"
#include "L1Trigger/RPCTrigger/src/L1RpcConst.h"

#include "L1Trigger/RPCTrigger/src/RPCException.h"

#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"


/** 
 *
 * Constructor required by L1RpcPacManager.
 * @param patFilesDir -  the directory conataing PAC definition file.
 * It should containe file for this PAC, defined by tower, logSector, logSegment,
 * named pacPat_t<tower>sc<logSector>sg<logSegment>.vhd
 * Containers: EnergeticPatternsGroupList and TrackPatternsGroup are
 * filled with patterns from file (the method ParsePatternFile() is called).
 * 
 */
L1RpcPac::L1RpcPac(std::string patFilesDir, int tower, int logSector, int logSegment):
  L1RpcPacBase(tower, logSector, logSegment) { 
  
  std::string patFileName;
  
  L1RpcConst rpcconst;

//  if(patFilesDir.find("pat") != std::string::npos) {
    patFileName = patFilesDir 
        + "pacPat_t" + rpcconst.IntToString(ConeCrdnts.Tower) 
        + "sc" + rpcconst.IntToString(ConeCrdnts.LogSector) 
        + "sg" + rpcconst.IntToString(ConeCrdnts.LogSegment) 
        + ".xml";

    L1RpcPatternsParser parser;
    parser.Parse(patFileName);
    Init(parser);
//  }
/*
  else {
    throw L1RpcException("patFilesDir empty (no patterns)");
    //edm::LogError("RPCTrigger")<< "patFilesDir not containes XML";
  }
*/  

  TrackPatternsGroup.SetGroupDescription("Track PatternsGroup");
  TrackPatternsGroup.SetGroupDescription("Track PatternsGroup");
  
}
/**
 *
 * @return the count af all patterns gropu, i.e. 1 + EnergeticPatternsGroupList.size(). 
 *
 */
int L1RpcPac::GetPatternsGroupCount () {
  return (1 + EnergeticPatternsGroupList.size() ); //1 = track pattrens group
}
/**
 *
 * @return pattern stored in PatternsVec.
 * Needed for patterns explorer.
 *
 */
L1RpcPattern L1RpcPac::GetPattern(int patNum) const {
  if(PatternsVec.size() == 0) {
  
    throw L1RpcException("GetPattren(): Patterns vec is empty, mayby it was not filled!");
    //edm::LogError("RPCTrigger") << "GetPattren(): Patterns vec is empty, mayby it was not filled!";
    
  }
  return PatternsVec[patNum];
  
}
/**
 * 
 *@return the count of patterns stored in PatternsVec.
 *
 */
int L1RpcPac::GetPatternsCount() {
  return PatternsVec.size();
}
/**
 *
 *@return true, if logStrip defined by logStripNum and logPlane  belongs to the
 * TrackPatternsGroup. 
 *
*/
bool L1RpcPac::GetTPatternsGroupShape(int logPlane, int logStripNum) {
  return TrackPatternsGroup.GroupShape.GetLogStripState(logPlane, logStripNum);
}
/** 
 *
 * @return true, if logStrip defined by logStripNum and logPlane  belongs to the
 * EPatternsGroup from EnergeticPatternsGroupList defined by groupNum. 
 * 
*/
bool L1RpcPac::GetEPatternsGroupShape(int groupNum, int logPlane, int bitNum) {
  TEPatternsGroupList::const_iterator iEGroup = EnergeticPatternsGroupList.begin();
  int i = 0;
  for(; iEGroup != EnergeticPatternsGroupList.end(); iEGroup++, i++) {
    if(i == groupNum)
      return iEGroup->GroupShape.GetLogStripState(logPlane, bitNum);
  }
  throw L1RpcException("GetEPatternsGroupShape(): groupNum to big!");
  //edm::LogError("RPCTrigger")<< "GetEPatternsGroupShape(): groupNum to big!";
  return false; // XXX - TMF
}

std::string L1RpcPac::GetPatternsGroupDescription(int patternGroupNum) {
  
  L1RpcConst rpcconst;
  
  std::string ret;
  if(patternGroupNum == -1)
    ret =  TrackPatternsGroup.GetGroupDescription();
  else  {
    TEPatternsGroupList::iterator iEGroup = EnergeticPatternsGroupList.begin();
    int i = 0;
    for(; iEGroup != EnergeticPatternsGroupList.end(); iEGroup++, i++) {
      if(i == patternGroupNum)
        ret = "EGroup #"+ rpcconst.IntToString(i)+iEGroup->GetGroupDescription();
    }
    
  }
  
  if (ret.empty()){
      throw L1RpcException("GetEPatternsGroupShape(): groupNum to big!");
      //edm::LogError("RPCTrigger")<< "GetEPatternsGroupShape(): groupNum to big!";
  }

  return ret;
}

void L1RpcPac::InsertQualityRecord(unsigned int qualityTabNumber,
                              unsigned short firedPlanes, short quality) {
  if(quality > MaxQuality)
    MaxQuality = quality;
  if(qualityTabNumber < QualityTabsVec.size() ) {
    QualityTabsVec[qualityTabNumber][firedPlanes] = quality;                   
  }
  else if(qualityTabNumber == QualityTabsVec.size() ) {
    // XXX - added cast (int)
    L1RpcConst::TQualityTab qualityTab((int)std::pow(2.0,L1RpcConst::LOGPLANES_COUNT), -1); //= new TQualityTab();
    QualityTabsVec.push_back(qualityTab);
    QualityTabsVec[qualityTabNumber][firedPlanes] = quality; 
  }
  else
    throw L1RpcException("InsertQualityRecord(): wrong qualityTabNumber");
    //edm::LogError("RPCTrigger") << "InsertQualityRecord(): wrong qualityTabNumber";
};


void L1RpcPac::InsertPatterns(const L1RpcPatternsVec& patternsVec) {
     
  L1RpcConst rpcconst;
  
  for(L1RpcPatternsVec::const_iterator patIt = patternsVec.begin(); patIt != patternsVec.end(); patIt++) { 
    if(patIt->GetPatternType() == L1RpcConst::PAT_TYPE_T)
      TrackPatternsGroup.AddPattern(patIt);
    else if (patIt->GetPatternType() == L1RpcConst::PAT_TYPE_E) {
      TEPatternsGroupList::iterator iEGroup;
      for(iEGroup = EnergeticPatternsGroupList.begin();
          iEGroup != EnergeticPatternsGroupList.end(); iEGroup++)
        if(iEGroup->Check(patIt) )
          break;
      if(iEGroup == EnergeticPatternsGroupList.end() ) {
        TEPatternsGroup newEGroup(patIt);
        newEGroup.SetGroupDescription(
        //"EGroup #"+ rpcconst.IntToString(EnergeticPatternsGroupList.size())+
        ", code: " + rpcconst.IntToString(patIt->GetCode()) +
        ", dir: " + rpcconst.IntToString(patIt->GetSign()) +
        ", refGroup: " + rpcconst.IntToString(patIt->GetRefGroup()) +
        ", qualityTabNumber: " + rpcconst.IntToString(patIt->GetQualityTabNumber()) );
        EnergeticPatternsGroupList.push_back(newEGroup);
      }
      else
       iEGroup->AddPattern(patIt);
    }
    else
      throw L1RpcException("InsertPattern(): unsupported pattern type");
      //edm::LogError("RPCTrigger") << "InsertPattern(): unsupported pattern type";
  }  

  if(EnergeticPatternsGroupList.size() != 0) {
    EnergeticPatternsGroupList.sort();           //to jest potrzebne, bo w Run() przechodzi                                                                         //pierwszy paettern, ktory ma Maxymalna quality, wiec grupy musza byc 
    EnergeticPatternsGroupList.reverse();
  } 
};

void L1RpcPac::Init(const L1RpcPatternsParser& parser) {
  for(unsigned int i = 0; i < parser.GetQualityVec().size(); i++) {    
    L1RpcPatternsParser::TQuality quality = parser.GetQualityVec()[i];
    std::bitset<L1RpcConst::LOGPLANES_COUNT> qualBits(quality.FiredPlanes );
    unsigned short firedPlanes = qualBits.to_ulong();

    InsertQualityRecord(quality.QualityTabNumber, firedPlanes, quality.QualityValue);  
  }

  PatternsVec = parser.GetPatternsVec(ConeCrdnts);
  InsertPatterns(PatternsVec);
}

L1RpcPacMuon L1RpcPac::RunTrackPatternsGroup(const L1RpcLogCone& cone) const {
  L1RpcPacMuon bestMuon;

  for(unsigned int vecNum = 0; vecNum < TrackPatternsGroup.PatternsItVec.size(); vecNum++) {
    unsigned short firedPlanes = 0;
    int firedPlanesCount = 0;
    unsigned short one = 1;
    const L1RpcPattern& pattern  = *(TrackPatternsGroup.PatternsItVec[vecNum]);
    for(int logPlane = L1RpcConst::FIRST_PLANE; logPlane < L1RpcConst::USED_PLANES_COUNT[ConeCrdnts.Tower]; logPlane++) {
      if (pattern.GetStripFrom(logPlane) == L1RpcConst::NOT_CONECTED) {
        //firedPlanes[logPlane] = false; //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        continue;
      }
      int fromBit = pattern.GetStripFrom(logPlane);
      int toBit = pattern.GetStripTo(logPlane);
      for(int bitNumber = fromBit; bitNumber < toBit; bitNumber++) {
        if(cone.GetLogStripState(logPlane, bitNumber) == true) {
          firedPlanes  = firedPlanes | one;
          firedPlanesCount++;
          break;
        }              
      }
 
      if( (L1RpcConst::USED_PLANES_COUNT[ConeCrdnts.Tower] - logPlane) == 3)
        if( firedPlanesCount == 0)
          break;
      
      one = one<<1;
    }

    if(firedPlanesCount >= 3) {
      short quality = QualityTabsVec[pattern.GetQualityTabNumber()][firedPlanes];
      if(quality != -1) {
        if(quality >= bestMuon.GetQuality() ) {
          L1RpcPacMuon bufMuon(pattern, quality, firedPlanes);
          if(bufMuon > bestMuon) {

            bestMuon = bufMuon;
            //if(bestMuon.GetQuality() == MaxQuality ) //it can not be if there are patts of both sign sorted by sign
            //  return bestMuon;
          }
        }
      }
    }
  }
  return bestMuon;
};

L1RpcPacMuon L1RpcPac::RunEnergeticPatternsGroups(const L1RpcLogCone& cone) const {
  L1RpcPacMuon bestMuon;
  unsigned short firedPlanes = 0;
  int firedPlanesCount = 0;
  TEPatternsGroupList::const_iterator iEGroup = EnergeticPatternsGroupList.begin();
  for(; iEGroup != EnergeticPatternsGroupList.end(); iEGroup++) {
    firedPlanes = 0;
    firedPlanesCount = 0;
    unsigned short one = 1;
    for(int logPlane = L1RpcConst::FIRST_PLANE; logPlane < L1RpcConst::USED_PLANES_COUNT[ConeCrdnts.Tower]; logPlane++) {  //or po paskach ze stozka
      for(unsigned int bitNum = 0; bitNum < L1RpcConst::LOGPLANE_SIZE[abs(ConeCrdnts.Tower)][logPlane] ; bitNum++) {       
        if(iEGroup->GroupShape.GetLogStripState(logPlane, bitNum) && cone.GetLogStripState(logPlane, bitNum) ) {
          firedPlanes  = firedPlanes | one;
          firedPlanesCount++;
          break;     
        }
      }
      one = one << 1;
    }

    short quality = QualityTabsVec[iEGroup->QualityTabNumber][firedPlanes];
    if(quality == -1)
      continue;

    L1RpcPacMuon bufMuon;
    for(unsigned int vecNum = 0; vecNum < iEGroup->PatternsItVec.size(); vecNum++) {
      const L1RpcPatternsVec::const_iterator patternIt = iEGroup->PatternsItVec[vecNum];      
      const L1RpcPattern& pattern = *patternIt;     
      bool wasHit = false;
      unsigned short one1 = 1;
      for(int logPlane = L1RpcConst::FIRST_PLANE; logPlane < L1RpcConst::USED_PLANES_COUNT[ConeCrdnts.Tower]; logPlane++, one1 = one1<<1) {        
        if (pattern.GetStripFrom(logPlane) == L1RpcConst::NOT_CONECTED) {
//          firedPlanes[logPlane] = false; //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
          continue;
        }
        if((firedPlanes & one1) != 0) {
          int fromBit = pattern.GetStripFrom(logPlane);
          int toBit = pattern.GetStripTo(logPlane);
          wasHit = false;
          for(int bitNumber = fromBit; bitNumber < toBit; bitNumber++)
            wasHit = wasHit || cone.GetLogStripState(logPlane, bitNumber);
          if(!wasHit)
            break;
         }      
      }
      if(wasHit) {
        bufMuon.SetAll(pattern, quality, firedPlanes);
        break;//if one pattern fits, thers no point to check other patterns from group
      }
    } //end of patterns loop
    if(bufMuon > bestMuon)
      bestMuon = bufMuon;
    if(bestMuon.GetQuality() == MaxQuality)
      return bestMuon;
  }//end of EGroup loop
  return bestMuon;
}
/** 
 *
 * Performs Pattern Comparator algorithm for hits from the cone.
 * Calls the RunTrackPatternsGroup() and RunEnergeticPatternsGroups().
 * @return found track candidate (empty if hits does not fit to eny pattern)
 *
*/
L1RpcPacMuon L1RpcPac::Run(const L1RpcLogCone& cone) const {  //symualcja
  //track
  L1RpcPacMuon bestMuon;
  if(TrackPatternsGroup.PatternsItVec.size() > 0)
    bestMuon = RunTrackPatternsGroup(cone);

  //energetic
  if(EnergeticPatternsGroupList.size() > 0) {
    L1RpcPacMuon bufMuon = RunEnergeticPatternsGroups(cone);
    if(bufMuon > bestMuon)
      bestMuon = bufMuon;
  }

  bestMuon.SetConeCrdnts(CurrConeCrdnts);
  bestMuon.SetLogConeIdx(cone.GetIdx());
  int refStripNum = GetPattern(bestMuon.GetPatternNum()).GetStripFrom(L1RpcConst::REF_PLANE[abs(CurrConeCrdnts.Tower)]) + CurrConeCrdnts.LogSector * 96 + CurrConeCrdnts.LogSegment * 8;
	bestMuon.SetRefStripNum(refStripNum);
  return bestMuon;
};
//------------------------------------------------------------------------------



