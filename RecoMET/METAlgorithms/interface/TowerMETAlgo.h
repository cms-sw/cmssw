#ifndef TowerMETAlgo_h
#define TowerMETAlgo_h

/** \class TowerMETAlgo
 *
 * Calculates MET for given input CaloTower collection.
 * Does corrections based on supplied parameters.
 *
 * \author M. Schmitt, R. Cavanaugh, The University of Florida
 *
 * \version   1st Version May 14, 2005
 ************************************************************/

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h" 
#include "DataFormats/METObjects/interface/TowerMETCollection.h"

class TowerMETAlgo 
{
 public:
  TowerMETAlgo();
  virtual ~TowerMETAlgo();
  virtual void run(const CaloTowerCollection *towers, TowerMETCollection &);
 private:
};

#endif // TowerMETAlgo_h

/*  LocalWords:  TowerMETAlgo
 */
