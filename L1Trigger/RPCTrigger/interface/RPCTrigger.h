#ifndef RPCTrigger_RPCTrigger_h
#define RPCTrigger_RPCTrigger_h

/** \class RPCTrigger
 *  \brief Implements RPC trigger emulation
 *
 *  $Date: 2006/05/30 18:48:39 $
 *  $Revision: 1.3 $
 *  \author Tomasz Fruboes
 *  \todo Give output in a kosher way
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

// L1RpcTrigger specific includes
#include "L1Trigger/RPCTrigger/src/RPCTriggerGeo.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPacManager.h"

#include "L1Trigger/RPCTrigger/src/L1RpcPacTrigger.h"
#include "L1Trigger/RPCTrigger/src/L1RpcBasicTrigConfig.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPac.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPacManager.h"

//class RPCTriggerGeo;

class RPCTrigger : public edm::EDProducer {
  public:
    explicit RPCTrigger(const edm::ParameterSet&);
    ~RPCTrigger();


    virtual void produce(edm::Event&, const edm::EventSetup&);
  private:
      // ----------member data ---------------------------
    
    
    RPCTriggerGeo theLinksystem;  ///< Tells where to send no of fired strip.
    
    L1RpcPacManager<L1RpcPac> m_pacManager;
    
    L1RpcBasicTrigConfig* m_trigConfig;
    
    L1RpcPacTrigger* m_pacTrigger;
};


#endif
