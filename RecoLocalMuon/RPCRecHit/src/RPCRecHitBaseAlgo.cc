/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/10/14 09:12:38 $
 *  $Revision: 1.6 $
 *  \author M. Maggi -- INFN Bari
 */



#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCClusterContainer.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCCluster.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCClusterizer.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
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
							 const RPCDigiCollection::Range& digiRange,
                                                         const RollMask& mask) {
  edm::OwnVector<RPCRecHit> result; 


  RPCClusterizer clizer;
  RPCClusterContainer tcls = clizer.doAction(digiRange);
  RPCMaskReClusterizer mrclizer;
  RPCClusterContainer cls = mrclizer.doAction(rpcId,tcls,mask);



//   // Reporting on ReClusterization

//   if ( (tcls.size() - cls.size()) != 0 ) {
//     std::cout << std::endl;
//     std::cout << "%%%%% RECLUSTERIZATION TAKING PLACE %%%%%" << std::endl << std::endl;
//     std::cout << "tcls container size : " << tcls.size() << std::endl;
//     std::cout << "cls container size  : " << cls.size() << std::endl;
//     RPCGeomServ Reporter(rpcId);
//     std::cout << "RPCDetId :  " << Reporter.name() << std::endl << std::endl;
//     std::cout << "Mask :" << std::endl << std::endl;
//     for (int i = 0; i < 96; i++) {
//       if (mask.test(i)) std::cout << "1";
//       else std::cout <<"0"; 
//     }
//     std::cout << std::endl;

//     std::cout << std::endl;
//     std::cout << "***  Original Clusters : " << std::endl << std::endl;

//     int j = 1;
//     for (RPCClusterContainer::const_iterator i = tcls.begin(); i != tcls.end(); i++ ) {
//       RPCCluster cl = *i;
//       std::cout << "   Cluster # " << j;
//       std::cout << "   Bunch crossing : " << cl.bx();
//       std::cout << "   First strip : " << cl.firstStrip();
//       std::cout << "   Last strip : " << cl.lastStrip();
//       std::cout << std::endl;
//       j++;
//     }

//     std::cout << std::endl;
//     std::cout << "*** New Clusters : " << std::endl << std::endl;

//     j = 1;
//     for (RPCClusterContainer::const_iterator i = cls.begin(); i != cls.end(); i++ ) {
//       RPCCluster cl = *i;
//       std::cout << "   Cluster # " << j;
//       std::cout << "   Bunch crossing : " << cl.bx();
//       std::cout << "   First strip : " << cl.firstStrip();
//       std::cout << "   Last strip : " << cl.lastStrip();
//       std::cout << std::endl;
//       j++;
//     }

//     std::cout << std::endl;
//   }                           // End of report on ReClusterization


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


    result.push_back(recHit);
  }
  return result;
}
