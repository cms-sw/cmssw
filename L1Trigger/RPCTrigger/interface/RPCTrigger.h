#ifndef L1Trigger_RPCTrigger_h
#define L1Trigger_RPCTrigger_h

/** \class RPCTrigger
 *  \brief Implements RPC trigger emulation
 *
 *  \author Tomasz Fruboes
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"


#include <FWCore/Framework/interface/ESHandle.h> // Handle to read geometry
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"


// L1RpcTrigger specific includes
#include "L1Trigger/RPCTrigger/interface/RPCConeBuilderFromES.h"

#include "L1Trigger/RPCTrigger/interface/RPCPacManager.h"

#include "L1Trigger/RPCTrigger/interface/RPCPacTrigger.h"
#include "L1Trigger/RPCTrigger/interface/RPCBasicTrigConfig.h"
#include "L1Trigger/RPCTrigger/interface/RPCPacData.h"
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
#include "L1Trigger/RPCTrigger/interface/RPCPacManager.h"
#include "CondFormats/DataRecord/interface/L1RPCHsbConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"
#include "DataFormats/RPCDigi/interface/RPCDigiL1Link.h"
#include <memory>
#include <vector>



//class RPCTriggerGeo;

class RPCTrigger : public edm::EDProducer {
  public:
    explicit RPCTrigger(const edm::ParameterSet&);
    ~RPCTrigger();


    virtual void produce(edm::Event&, const edm::EventSetup&);
  private:
      // ----------member data ---------------------------
    
    
    RPCConeBuilderFromES m_theLinksystemFromES;

    RPCPacManager<RPCPacData> m_pacManager;
    
    RPCBasicTrigConfig* m_trigConfig;
    
    RPCPacTrigger* m_pacTrigger;
 
    bool m_firstRun;   
    int m_triggerDebug;
    unsigned long long m_cacheID;
    // TODO keep L1MuRegionalCandVec equally as RPCDigiL1LinkVec
    std::vector<L1MuRegionalCand> giveFinallCandindates(const L1RpcTBMuonsVec& finalMuons, int type, int bx,   
                                     edm::Handle<RPCDigiCollection> rpcDigis, std::vector<RPCDigiL1Link> & retRPCDigiLink);

    std::string m_label;

};


#endif
