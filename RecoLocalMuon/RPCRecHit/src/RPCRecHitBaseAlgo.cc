/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/04/12 10:22:00 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */



#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCClusterContainer.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCCluster.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCClusterizer.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


RPCRecHitBaseAlgo::RPCRecHitBaseAlgo(const edm::ParameterSet& config) {
  //  theSync = RPCTTrigSyncFactory::get()->create(config.getParameter<string>("tTrigMode"),
  //config.getParameter<ParameterSet>("tTrigModeConfig"));
}

RPCRecHitBaseAlgo::~RPCRecHitBaseAlgo(){}


// Build all hits in the range associated to the layerId, at the 1st step.
edm::OwnVector<RPCRecHit> RPCRecHitBaseAlgo::reconstruct(const RPCRoll& roll,
							 const RPCDetId& rpcId,
							 const RPCDigiCollection::Range& digiRange) {
  edm::OwnVector<RPCRecHit> result; 

  // Loop over all digis in the given range and fill the clusterizer
  // with digis that will represent the seed...
  RPCClusterContainer cls;
  for (RPCDigiCollection::const_iterator digi = digiRange.first;
       digi != digiRange.second;
       digi++) {
    RPCCluster cl(digi->strip(),digi->strip(),digi->bx());
    cls.insert(cl);
  }
  RPCClusterizer clizer;
  // This will eventually modify the RPCCLusterContainer content..
  clizer.doAction(cls);
  for (RPCClusterContainer::const_iterator cl = cls.begin();
       cl != cls.end(); cl++){
    
    LocalError tmpErr;
    LocalPoint point;
    // Call the compute method
    bool OK = this->compute(roll, *cl, point, tmpErr);
    if (!OK) continue;

    // Build a new pair of 1D rechit    
    RPCRecHit*  recHit = new RPCRecHit(rpcId,cl->bx(),point,tmpErr);


    result.push_back(recHit);
  }
  return result;
}
