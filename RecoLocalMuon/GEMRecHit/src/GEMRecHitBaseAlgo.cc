/*
 *  See header file for a description of this class.
 *
 *  $Date: 2013/04/24 17:16:35 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
 */



#include "RecoLocalMuon/GEMRecHit/interface/GEMRecHitBaseAlgo.h"
#include "RecoLocalMuon/GEMRecHit/src/GEMClusterContainer.h"
#include "RecoLocalMuon/GEMRecHit/src/GEMCluster.h"
#include "RecoLocalMuon/GEMRecHit/src/GEMClusterizer.h"
#include "RecoLocalMuon/GEMRecHit/src/GEMMaskReClusterizer.h"

#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


GEMRecHitBaseAlgo::GEMRecHitBaseAlgo(const edm::ParameterSet& config):
  stationToUse_(config.getUntrackedParameter<int>("stationToUse",3))
{
  //  theSync = GEMTTrigSyncFactory::get()->create(config.getParameter<string>("tTrigMode"),
  //config.getParameter<ParameterSet>("tTrigModeConfig"));
}

GEMRecHitBaseAlgo::~GEMRecHitBaseAlgo(){}


// Build all hits in the range associated to the layerId, at the 1st step.
edm::OwnVector<GEMRecHit> GEMRecHitBaseAlgo::reconstruct(const GEMEtaPartition& roll,
							 const GEMDetId& gemId,
							 const GEMDigiCollection::Range& digiRange,
                                                         const EtaPartitionMask& mask) {
  edm::OwnVector<GEMRecHit> result; 


  GEMClusterizer clizer;
  GEMClusterContainer tcls = clizer.doAction(digiRange);
  GEMMaskReClusterizer mrclizer;
  GEMClusterContainer cls = mrclizer.doAction(gemId,tcls,mask);


  for (GEMClusterContainer::const_iterator cl = cls.begin();
       cl != cls.end(); cl++){
    
    LocalError tmpErr;
    LocalPoint point;
    // Call the compute method
    bool OK = this->compute(roll, *cl, point, tmpErr);
    if (!OK) continue;

    // Build a new pair of 1D rechit 
    int firstClustStrip= cl->firstStrip();
    int clusterSize=cl->clusterSize(); 
    GEMRecHit*  recHit = new GEMRecHit(gemId,cl->bx(),firstClustStrip,clusterSize,point,tmpErr);

    int station = (int) gemId.station();
    switch(stationToUse_){

	case 0: if(station != 1 && station != 2 && station != 3) result.push_back(recHit); //NO GEM
		break;
	case 1: if(station != 2 && station != 3) result.push_back(recHit); // NO GE2/1
		break;
	case 2: if(station != 1) result.push_back(recHit); // NO GE1/1
		break;
	case 3: result.push_back(recHit); // GE1/1 + GE2/1
		break;
	default: result.push_back(recHit); // GE1/1 + GE2/1
		break;

    }

    //result.push_back(recHit);
  }
  return result;
}
