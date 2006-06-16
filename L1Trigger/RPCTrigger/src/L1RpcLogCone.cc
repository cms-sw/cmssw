/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/L1RpcLogCone.h" 

//using namespace std;

L1RpcLogCone::L1RpcLogCone(const L1RpcLogHit &logHit) {
  LogPlanesVec.assign(RPCParam::LOGPLANES_COUNT, TLogPlane() );

  ConeCrdnts = logHit.GetConeCrdnts();

  MuonCode = 0;
  MuonSign = 0;

  SetLogStrip(logHit.getlogPlaneNumber() -1, logHit.getStripNumberInCone(), logHit.getDigiIdx());
}

bool L1RpcLogCone::AddLogHit(const L1RpcLogHit &logHit) {
  if (ConeCrdnts.Tower == logHit.getTower() &&
      ConeCrdnts.LogSector == logHit.getLogSector() &&
      ConeCrdnts.LogSegment == logHit.getLogSegment()  ) {
    SetLogStrip(logHit.getlogPlaneNumber()-1, logHit.getStripNumberInCone(), logHit.getDigiIdx());
    return true;
  }
  else
    return false;
}


void L1RpcLogCone::SetLogStrip(int logPlane, int logStripNum, int digiIdx) {
  //LogPlanesVec[logPlane].insert(logStripNum);
  //LogPlanesVec[logPlane].insert(TLogPlane::value_type(logStripNum, vector<int>() ) );
  LogPlanesVec[logPlane][logStripNum].push_back(digiIdx);
}

void L1RpcLogCone::SetLogStrip(int logPlane, int logStripNum) {
  LogPlanesVec[logPlane].insert(TLogPlane::value_type(logStripNum, vector<int>() ) );
}


bool L1RpcLogCone::GetLogStripState(int logPlane, unsigned int logStripNum)  const {
  return LogPlanesVec[logPlane].count(logStripNum);
}

vector<int> L1RpcLogCone::GetLogStripDigisIdxs(int logPlane, unsigned int logStripNum) const {
  TLogPlane::const_iterator it = LogPlanesVec[logPlane].find(logStripNum); 
  if(it != LogPlanesVec[logPlane].end() )
    return it->second;
  else
    return vector<int>();
}


bool L1RpcLogCone::IsPlaneFired(int logPlane) const {
  if(LogPlanesVec[logPlane].size() == 0)
    return false; 
  else
    return true;  
}

int L1RpcLogCone::GetFiredPlanesCnt() const{
  int firedPlanes = 0;
  for(int logPlane = RPCParam::FIRST_PLANE; logPlane < RPCParam::USED_PLANES_COUNT[abs(ConeCrdnts.Tower)]; logPlane++) {
    firedPlanes = firedPlanes + IsPlaneFired(logPlane);
  }
  return firedPlanes;
}


int L1RpcLogCone::PossibleTrigger() const {
  int triggerType = 0; //0 - trigger not possible
                        //1 - 3/4 (3 inner planes fired)
                        //2 - 4/6 (four palnes fired)
  int firedPlanes = 0;

  int logPlane = RPCParam::FIRST_PLANE;
  for( ; logPlane <= RPCParam::LOGPLANE4; logPlane++) {
    firedPlanes = firedPlanes + IsPlaneFired(logPlane);
  }
  if(firedPlanes >= 3)
    triggerType = 1;

  for( ; logPlane < RPCParam::USED_PLANES_COUNT[abs(ConeCrdnts.Tower)]; logPlane++) {
    firedPlanes = firedPlanes + IsPlaneFired(logPlane);
  }
  if(firedPlanes >= 4)
    triggerType = 2;

  return triggerType;
}
