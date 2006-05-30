#ifndef RPCTrigger_RPCTrigger_h
#define RPCTrigger_RPCTrigger_h

/** \class RPCTrigger
 *  \brief Implements RPC trigger emulation
 *
 *  $Date: 2006/05/29 12:00:00 $
 *  $Revision: 1.1 $
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


//class RPCTriggerGeo;

class RPCTrigger : public edm::EDProducer {
  public:
    explicit RPCTrigger(const edm::ParameterSet&);
    ~RPCTrigger();


    virtual void produce(edm::Event&, const edm::EventSetup&);
  private:
      // ----------member data ---------------------------
    
    
    RPCTriggerGeo theLinksystem;  ///< Tells where to send no of fired strip.
};


#endif
