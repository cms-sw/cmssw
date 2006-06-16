/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
//#include "Utilities/Configuration/interface/Architecture.h"
#include <math.h>
#include <bitset>
#include "L1Trigger/RPCTrigger/src/L1RpcPac.h"
//#include "L1Trigger/RPCTrigger/src/L1RpcParametersDef.h"
#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"
#ifndef _STAND_ALONE
//#include "L1Trigger/RPCTrigger/interface/L1Rpc.h"
//#include "Utilities/Notification/interface/Singleton.h"
#endif
#include <iostream>
using namespace std;

void L1RpcPac::TPatternsGroup::UpdateShape(const L1RpcPatternsVec::const_iterator& pattern) { //colled by AddPattern
  for(int logPlane = RPCParam::FIRST_PLANE; logPlane <= RPCParam::LAST_PLANE; logPlane++) {
    if (pattern->GetStripFrom(logPlane) != RPCParam::NOT_CONECTED) {
      int fromBit = pattern->GetStripFrom(logPlane);
      int toBit = pattern->GetStripTo(logPlane);
      for (int bitNumber = fromBit; bitNumber < toBit; bitNumber++)
        GroupShape.SetLogStrip(logPlane, bitNumber);
    }
  }
}

bool L1RpcPac::TEPatternsGroup::Check(const L1RpcPatternsVec::const_iterator& pattern) {
  if(PatternsItVec[0]->GetRefGroup() == pattern->GetRefGroup() &&
     PatternsItVec[0]->GetCode() == pattern->GetCode() &&
     PatternsItVec[0]->GetSign() == pattern->GetSign() &&
     PatternsItVec[0]->GetQualityTabNumber() == pattern->GetQualityTabNumber() )
    return true;
  return false;
}


bool L1RpcPac::TEPatternsGroup::operator < (const TEPatternsGroup& ePatternsGroup) const {
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
//------------------------------------------------------------------------------

bool L1RpcPac::GetEPatternsGroupShape(int groupNum, int logPlane, int bitNum) {
  TEPatternsGroupList::const_iterator iEGroup = EnergeticPatternsGroupList.begin();
  int i = 0;
  for(; iEGroup != EnergeticPatternsGroupList.end(); iEGroup++, i++) {
    if(i == groupNum)
      return iEGroup->GroupShape.GetLogStripState(logPlane, bitNum);
  }
  //throw L1RpcException("GetEPatternsGroupShape(): groupNum to big!");
  std::cout << "GetEPatternsGroupShape(): groupNum to big!" << std::endl;
}

std::string L1RpcPac::GetPatternsGroupDescription(int patternGroupNum) {
  if(patternGroupNum == -1)
    return  TrackPatternsGroup.GetGroupDescription();
  else  {
    TEPatternsGroupList::iterator iEGroup = EnergeticPatternsGroupList.begin();
    int i = 0;
    for(; iEGroup != EnergeticPatternsGroupList.end(); iEGroup++, i++) {
      if(i == patternGroupNum)
        return "EGroup #"+ RPCParam::IntToString(i)+iEGroup->GetGroupDescription();
    }
    //throw L1RpcException("GetEPatternsGroupShape(): groupNum to big!");
    std::cout<< "GetEPatternsGroupShape(): groupNum to big!" << std::endl; 
  }
}

void L1RpcPac::InsertQualityRecord(int qualityTabNumber,
                              unsigned short firedPlanes, short quality) {
  if(quality > MaxQuality)
    MaxQuality = quality;
  if(qualityTabNumber < QualityTabsVec.size() ) {
    QualityTabsVec[qualityTabNumber][firedPlanes] = quality;                   
  }
  else if(qualityTabNumber == QualityTabsVec.size() ) {
    RPCParam::TQualityTab qualityTab(pow(2.0,RPCParam::LOGPLANES_COUNT), -1); //= new TQualityTab();
    QualityTabsVec.push_back(qualityTab);
    QualityTabsVec[qualityTabNumber][firedPlanes] = quality; 
  }
  else
    //throw L1RpcException("InsertQualityRecord(): wrong qualityTabNumber");
    std::cout << "InsertQualityRecord(): wrong qualityTabNumber" << std::endl;
};


void L1RpcPac::InsertPatterns(const L1RpcPatternsVec& patternsVec) {
  for(L1RpcPatternsVec::const_iterator patIt = patternsVec.begin(); patIt != patternsVec.end(); patIt++) { 
    if(patIt->GetPatternType() == RPCParam::PAT_TYPE_T)
      TrackPatternsGroup.AddPattern(patIt);
    else if (patIt->GetPatternType() == RPCParam::PAT_TYPE_E) {
      TEPatternsGroupList::iterator iEGroup;
      for(iEGroup = EnergeticPatternsGroupList.begin();
          iEGroup != EnergeticPatternsGroupList.end(); iEGroup++)
        if(iEGroup->Check(patIt) )
          break;
      if(iEGroup == EnergeticPatternsGroupList.end() ) {
        TEPatternsGroup newEGroup(patIt);
        newEGroup.SetGroupDescription(
        //"EGroup #"+ RPCParam::IntToString(EnergeticPatternsGroupList.size())+
        ", code: " + RPCParam::IntToString(patIt->GetCode()) +
        ", dir: " + RPCParam::IntToString(patIt->GetSign()) +
        ", refGroup: " + RPCParam::IntToString(patIt->GetRefGroup()) +
        ", qualityTabNumber: " + RPCParam::IntToString(patIt->GetQualityTabNumber()) );
        EnergeticPatternsGroupList.push_back(newEGroup);
      }
      else
       iEGroup->AddPattern(patIt);
    }
    else
      //throw L1RpcException("InsertPattern(): unsupported pattern type");
      std::cout << "InsertPattern(): unsupported pattern type" << std::endl;
  }  

  if(EnergeticPatternsGroupList.size() != 0) {
    EnergeticPatternsGroupList.sort();           //to jest potrzebne, bo w Run() przechodzi                                                                         //pierwszy paettern, ktory ma Maxymalna quality, wiec grupy musza byc 
    EnergeticPatternsGroupList.reverse();
  } 
};

void L1RpcPac::Init(const L1RpcPatternsParser& parser) {
  for(unsigned int i = 0; i < parser.GetQualityVec().size(); i++) {    
    L1RpcPatternsParser::TQuality quality = parser.GetQualityVec()[i];
    bitset<RPCParam::LOGPLANES_COUNT> qualBits(quality.FiredPlanes );
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
    for(int logPlane = RPCParam::FIRST_PLANE; logPlane < RPCParam::USED_PLANES_COUNT[ConeCrdnts.Tower]; logPlane++) {
      if (pattern.GetStripFrom(logPlane) == RPCParam::NOT_CONECTED) {
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
 
      if( (RPCParam::USED_PLANES_COUNT[ConeCrdnts.Tower] - logPlane) == 3)
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
    for(int logPlane = RPCParam::FIRST_PLANE; logPlane < RPCParam::USED_PLANES_COUNT[ConeCrdnts.Tower]; logPlane++) {  //or po paskach ze stozka
      for(int bitNum = 0; bitNum < RPCParam::LOGPLANE_SIZE[abs(ConeCrdnts.Tower)][logPlane] ; bitNum++) {       
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
      for(int logPlane = RPCParam::FIRST_PLANE; logPlane < RPCParam::USED_PLANES_COUNT[ConeCrdnts.Tower]; logPlane++, one1 = one1<<1) {        
        if (pattern.GetStripFrom(logPlane) == RPCParam::NOT_CONECTED) {
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
};

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
  int refStripNum = GetPattern(bestMuon.GetPatternNum()).GetStripFrom(RPCParam::REF_PLANE[abs(CurrConeCrdnts.Tower)]) + CurrConeCrdnts.LogSector * 96 + CurrConeCrdnts.LogSegment * 8;
	bestMuon.SetRefStripNum(refStripNum);
  return bestMuon;
};
//------------------------------------------------------------------------------



