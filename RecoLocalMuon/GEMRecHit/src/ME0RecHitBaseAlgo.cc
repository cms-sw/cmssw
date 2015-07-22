/*
* See header file for a description of this class.
*
* $Date: 2014/02/04 10:16:35 $
* $Revision: 1.1 $
* \author M. Maggi -- INFN Bari
*/



#include "RecoLocalMuon/GEMRecHit/interface/ME0RecHitBaseAlgo.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


ME0RecHitBaseAlgo::ME0RecHitBaseAlgo(const edm::ParameterSet& config) {
}

ME0RecHitBaseAlgo::~ME0RecHitBaseAlgo(){}


// Build all hits in the range associated to the layerId, at the 1st step.
edm::OwnVector<ME0RecHit> ME0RecHitBaseAlgo::reconstruct(const ME0DetId& me0Id,
const ME0DigiPreRecoCollection::Range& digiRange){
  edm::OwnVector<ME0RecHit> result;

  for (ME0DigiPreRecoCollection::const_iterator digi = digiRange.first;
       digi != digiRange.second;digi++) {
    
    LocalError tmpErr;
    LocalPoint point;
    // Call the compute method
    bool OK = this->compute(*digi, point, tmpErr);
    if (!OK) continue;
   
    if (std::abs(digi->pdgid()) == 13) {
       ME0RecHit* recHit = new ME0RecHit(me0Id,digi->tof(),point,tmpErr);
       result.push_back(recHit);
    }

  }
  return result;
}
