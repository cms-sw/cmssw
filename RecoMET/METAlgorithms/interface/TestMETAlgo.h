#ifndef TestMETAlgo_h
#define TestMETAlgo_h

/** \class TestMETAlgo
 *
 * Calculates MET for given input CaloTower collection.
 * Does corrections based on supplied parameters.
 *
 * \author R. Cavanaugh, The University of Florida
 *
 * \version   1st Version May 14, 2005
 ************************************************************/

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h" //Added "Collection"
#include "DataFormats/METObjects/interface/METCollection.h"

class TestMETAlgo 
{
 public:
  TestMETAlgo(bool Type1);
  virtual ~TestMETAlgo();
  virtual void run(const CaloTowerCollection *towers, METCollection &);
 private:
  // Implementation Methods
  double calcMETPhi(double MEX, double MEY) const;
  // Data Members
  bool doType1;
};

#endif // TestMETAlgo_h

/*  LocalWords:  TestMETAlgo
 */
