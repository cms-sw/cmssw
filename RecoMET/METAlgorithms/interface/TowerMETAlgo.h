#ifndef TowerMETAlgo_h
#define TowerMETAlgo_h

/** \class TowerMETAlgo
 *
 * Calculates MET for given input CaloTower collection.
 * Does corrections based on supplied parameters.
 *
 * \author Michael Schmitt, The University of Florida
 *
 * \version   1st Version May 14, 2005
 ************************************************************/

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CaloObjects/interface/CaloTowerCollection.h"
#include "DataFormats/METObjects/interface/TowerMETCollection.h"
#include "DataFormats/JetObjects/interface/CaloJetCollection.h"

class TowerMETAlgo {

public:

  TowerMETAlgo(bool Type1);
  virtual ~TowerMETAlgo();

  virtual void run(const CaloTowerCollection *towers, 
                   const CaloJetCollection *rawjets,
                   const CaloJetCollection *corrjets,
                   TowerMETCollection &);

private:

  // Implementation Methods
  double calcMETPhi(double MEX, double MEY) const;

  // Data Members
  bool doType1;

};

#endif // TowerMETAlgo_h
