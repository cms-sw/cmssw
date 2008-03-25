#ifndef L1Trigger_RPCTrigger_h
#define L1Trigger_RPCTrigger_h

/** \class RPCTrigger
 *  \brief Implements RPC trigger emulation
 *
 *  $Date: 2007/06/06 13:39:13 $
 *  $Revision: 1.12 $
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

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"


// L1RpcTrigger specific includes
#include "L1Trigger/RPCTrigger/interface/RPCTriggerGeo.h"
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
    
    
    RPCTriggerGeo m_theLinksystem;  ///< Tells where to send no of fired strip.
    
    RPCPacManager<RPCPacData> m_pacManager;
    
    RPCBasicTrigConfig* m_trigConfig;
    
    RPCPacTrigger* m_pacTrigger;
 
    bool m_firstRun;   
    bool m_fixRPCGeo; 
    int m_triggerDebug;
    std::vector<L1MuRegionalCand> giveFinallCandindates(L1RpcTBMuonsVec finalMuons, short type);

    std::string m_label;

};


#endif
