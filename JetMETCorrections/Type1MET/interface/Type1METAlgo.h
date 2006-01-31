#ifndef Type1METAlgo_h
#define Type1METAlgo_h

/** \class Type1METAlgo
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
#include "DataFormats/METObjects/interface/METCollection.h"

class Type1METAlgo 
{
 public:
  Type1METAlgo();
  virtual ~Type1METAlgo();
  virtual void run(const METCollection*, METCollection &);
};

#endif // Type1METAlgo_h

/*  LocalWords:  Type1METAlgo
 */
