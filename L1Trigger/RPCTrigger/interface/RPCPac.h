#ifndef L1Trigger_RPCPac_h
#define L1Trigger_RPCPac_h

#include "L1Trigger/RPCTrigger/interface/RPCPacBase.h"
#include "L1Trigger/RPCTrigger/interface/RPCPacMuon.h"

#include "L1Trigger/RPCTrigger/interface/RPCLogCone.h"
#include "L1Trigger/RPCTrigger/interface/RPCPacData.h"

//class RPCLogCone;
//class RPCPacData;


class RPCPac: public RPCPacBase {
  
  public: 
    RPCPac(const RPCPacData *, int tower, int logSector, int logSegment);

    RPCPacMuon run(const RPCLogCone& cone) const;
    
  private:
    
    RPCPacMuon runTrackPatternsGroup(const RPCLogCone& cone) const;
    
    RPCPacMuon runEnergeticPatternsGroups(const RPCLogCone& cone) const;
    
    const RPCPacData* m_pacData;
};
  

#endif
