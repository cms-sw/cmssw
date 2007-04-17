#ifndef L1Trigger_RPCPac_h
#define L1Trigger_RPCPac_h

#include "L1Trigger/RPCTrigger/src/RPCPacBase.h"
#include "L1Trigger/RPCTrigger/src/RPCPacMuon.h"

#include "L1Trigger/RPCTrigger/src/RPCLogCone.h"
#include "L1Trigger/RPCTrigger/src/RPCPacData.h"

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
