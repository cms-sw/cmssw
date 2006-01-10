#ifndef ReadRecHitAlgorithm_h
#define ReadRecHitAlgorithm_h
# // -*-C++-*-
/** \class ReadRecHitAlgorithm
 *
 * ReadRecHitAlgorithm finds seeds
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Aug. 1, 2005
 *
 ************************************************************/

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"

class ReadRecHitAlgorithm 
{
 public:
  
  ReadRecHitAlgorithm(const edm::ParameterSet& conf);
  ~ReadRecHitAlgorithm();
  

  /// Runs the algorithm
    void run(const SiStripRecHit2DLocalPosCollection* input);

 private:

  edm::ParameterSet conf_;
};

#endif
