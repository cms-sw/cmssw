#ifndef L1Trigger_RPCTrigger_h
#define L1Trigger_RPCTrigger_h

/** \class RPCTrigger
 *  \brief Implements RPC trigger emulation
 *
 *  $Date: 2008/08/29 08:28:12 $
 *  $Revision: 1.15 $
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
    std::vector<L1MuRegionalCand> giveFinallCandindates(L1RpcTBMuonsVec finalMuons, int type, int bx);


    std::string m_label;

};


#endif
