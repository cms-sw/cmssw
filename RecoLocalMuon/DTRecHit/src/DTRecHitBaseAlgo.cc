/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/02/19 11:43:57 $
 *  $Revision: 1.5 $
 *  \author N. Amapane & G. Cerminara - INFN Torino
 */



#include "RecoLocalMuon/DTRecHit/interface/DTRecHitBaseAlgo.h"

#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;
using namespace edm;


DTRecHitBaseAlgo::DTRecHitBaseAlgo(const ParameterSet& config) {
  theSync = DTTTrigSyncFactory::get()->create(config.getParameter<string>("tTrigMode"),
					      config.getParameter<ParameterSet>("tTrigModeConfig"));
}

DTRecHitBaseAlgo::~DTRecHitBaseAlgo(){}


// Build all hits in the range associated to the layerId, at the 1st step.
OwnVector<DTRecHit1DPair> DTRecHitBaseAlgo::reconstruct(const DTLayer* layer,
							const DTLayerId& layerId,
							const DTDigiCollection::Range& digiRange) {
  OwnVector<DTRecHit1DPair> result; 

  // Loop over all digis in the given range
  for (DTDigiCollection::const_iterator digi = digiRange.first;
       digi != digiRange.second;
       digi++) {
    // Get the wireId
    DTWireId wireId(layerId, (*digi).wire());
    
    LocalError tmpErr;
    LocalPoint lpoint, rpoint;
    // Call the compute method
    bool OK = compute(layer, *digi, lpoint, rpoint, tmpErr);
    if (!OK) continue;

    // Build a new pair of 1D rechit    
    DTRecHit1DPair*  recHitPair = new DTRecHit1DPair(wireId, *digi);

    // Set the position and the error of the 1D rechits
    recHitPair->setPositionAndError(DTEnums::Left, lpoint, tmpErr);
    recHitPair->setPositionAndError(DTEnums::Right, rpoint, tmpErr);        

    result.push_back(recHitPair);
  }
  return result;
}



