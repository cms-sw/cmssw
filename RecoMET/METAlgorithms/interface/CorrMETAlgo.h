#ifndef CorrMETAlgo_h
#define CorrMETAlgo_h

/** \class CorrMETAlgo
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

//#include "DataFormats/CaloObjects/interface/CaloTowerCollection.h"
#include "DataFormats/METObjects/interface/METCollection.h"
//#include "DataFormats/JetObjects/interface/CaloJetCollection.h"

class CorrMETAlgo {

public:

  CorrMETAlgo(bool Type1);
  virtual ~CorrMETAlgo();

  virtual void run(const METCollection *rawmet, METCollection &);

private:

  // Implementation Methods
  double calcMETPhi(double MEX, double MEY) const;

  // Data Members
  bool doType1;

};

#endif // CorrMETAlgo_h

/*  LocalWords:  CorrMETAlgo
 */
