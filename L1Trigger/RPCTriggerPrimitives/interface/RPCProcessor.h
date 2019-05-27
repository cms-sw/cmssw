#ifndef L1Trigger_RPCTriggerPrimitives_RPCProcessor_h
#define L1Trigger_RPCTriggerPrimitives_RPCProcessor_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitAlgoFactory.h"

#include "CondFormats/DataRecord/interface/RPCMaskedStripsRcd.h"
#include "CondFormats/DataRecord/interface/RPCDeadStripsRcd.h"
#include "CondFormats/RPCObjects/interface/RPCMaskedStrips.h"
#include "CondFormats/RPCObjects/interface/RPCDeadStrips.h"
#include "CondFormats/Serialization/interface/Serializable.h"


#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <string> 
#include <sstream>
#include <utility>
#include <vector>
#include <boost/cstdint.hpp>

class RPCProcessor{
  
 public:
  
  explicit RPCProcessor();
  ~RPCProcessor();
  
  struct Map_structure {
    
    std::string linkboard_;
    std::string linkboard_ID;
    std::string chamber1_;
    std::string chamber2_;
    COND_SERIALIZABLE;     	
  };
  
  void Process(const edm::Event& iEvent,
	       const edm::EventSetup& iSetup,
	       const edm::EDGetToken& RPCDigiToken,
	       RPCRecHitCollection& primitivedigi,
               std::unique_ptr<RPCMaskedStrips>& theRPCMaskedStripsObj,
	       std::unique_ptr<RPCDeadStrips>& theRPCDeadStripsObj,
	       std::unique_ptr<RPCRecHitBaseAlgo>& theAlgo,
               std::map<std::string, std::string> LBName_ChamberID_Map_1, 
               std::map<std::string, std::string> LBID_ChamberID_Map_1, 
               std::map<std::string, std::string> LBName_ChamberID_Map_2, 
	       std::map<std::string, std::string> LBID_ChamberID_Map_2,
	       bool ApplyLinkBoardCut_, 
	       int LinkboardCut, 
               int ClusterSizeCut ) const;
  
  static edm::OwnVector<RPCRecHit> ApplyClusterSizeCut(const edm::OwnVector<RPCRecHit> recHits_, int ClusterSizeCut_);
  static bool ApplyLinkBoardCut(int NClusters, int LinkboardCut);
  
  std::vector<Map_structure> const & GetMapVector() const {return MapVec;}
  std::vector<Map_structure> MapVec;
  
  std::string GetStringBarrel(const int ring_, const int station_, const int sector_, const int layer_, const int subsector_, const int roll_) const;
  std::string GetStringEndCap(const int station_, const int ring_, const int chamberID_) const;
  
  
  COND_SERIALIZABLE;
  
 private:
   
};
#endif


