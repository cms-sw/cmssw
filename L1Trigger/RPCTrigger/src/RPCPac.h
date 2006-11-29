#ifndef L1RpcPacH
#define L1RpcPacH

#include "L1Trigger/RPCTrigger/src/RPCPacBase.h"
#include "L1Trigger/RPCTrigger/src/RPCPacMuon.h"

#include "L1Trigger/RPCTrigger/src/RPCLogCone.h"
#include "L1Trigger/RPCTrigger/src/RPCPacData.h"

//class RPCLogCone;
//class RPCPacData;


class RPCPac: public RPCPacBase {
  
  public: 
    RPCPac(const RPCPacData *, int, int, int);

    RPCPacMuon run(const RPCLogCone& cone) const;
    
  private:
    
    RPCPacMuon runTrackPatternsGroup(const RPCLogCone& cone) const;
    
    RPCPacMuon runEnergeticPatternsGroups(const RPCLogCone& cone) const;
    
    const RPCPacData* m_pacData;
};
  

#endif
