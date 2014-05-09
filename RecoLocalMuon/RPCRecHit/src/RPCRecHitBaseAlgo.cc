/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/12/04 16:13:16 $
 *  $Revision: 1.8 $
 *  \author M. Maggi -- INFN Bari
 */



#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCClusterContainer.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCCluster.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCClusterizer.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCMaskReClusterizer.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


RPCRecHitBaseAlgo::RPCRecHitBaseAlgo(const edm::ParameterSet& config):
  stationToUse_(config.getUntrackedParameter<int>("stationToUse",3))
{
  //  theSync = RPCTTrigSyncFactory::get()->create(config.getParameter<string>("tTrigMode"),
  //config.getParameter<ParameterSet>("tTrigModeConfig"));
}

RPCRecHitBaseAlgo::~RPCRecHitBaseAlgo(){}


// Build all hits in the range associated to the layerId, at the 1st step.
edm::OwnVector<RPCRecHit> RPCRecHitBaseAlgo::reconstruct(const RPCRoll& roll,
							 const RPCDetId& rpcId,
							 const RPCDigiCollection::Range& digiRange,
                                                         const RollMask& mask) {
  edm::OwnVector<RPCRecHit> result; 


  RPCClusterizer clizer;
  RPCClusterContainer tcls = clizer.doAction(digiRange);
  RPCMaskReClusterizer mrclizer;
  RPCClusterContainer cls = mrclizer.doAction(rpcId,tcls,mask);


  for (RPCClusterContainer::const_iterator cl = cls.begin();
       cl != cls.end(); cl++){
    
    LocalError tmpErr;
    LocalPoint point;
    // Call the compute method
    bool OK = this->compute(roll, *cl, point, tmpErr);
    if (!OK) continue;

    // Build a new pair of 1D rechit 
    int firstClustStrip= cl->firstStrip();
    int clusterSize=cl->clusterSize(); 
    RPCRecHit*  recHit = new RPCRecHit(rpcId,cl->bx(),firstClustStrip,clusterSize,point,tmpErr);

    int station = (int) rpcId.station();
    int ring = (int) rpcId.ring();

    switch(stationToUse_){

	case 0: if(!((station == 3 || station == 4) && ring == 1)) result.push_back(recHit); // NO RPC UPGRADE
		break;
	case 1: if(!(station == 4 && ring == 1)) result.push_back(recHit); // RE3/1
		break;
	case 2: if(!(station == 3 && ring == 1)) result.push_back(recHit); //RE4/1
		break;
	case 3: result.push_back(recHit); // RE3/1 + RE4/1
		break;
	default: result.push_back(recHit); // RE3/1 + RE4/1
		break;

    }

    //result.push_back(recHit);
  }
  return result;
}
