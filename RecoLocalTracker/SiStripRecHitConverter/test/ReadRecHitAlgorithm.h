#ifndef ReadRecHitAlgorithm_h
#define ReadRecHitAlgorithm_h
# // -*-C++-*-
/** \class ReadRecHitAlgorithm
 *
 * ReadRecHitAlgorithm reads rechits
 *
 * \author Chiara Genta
 *
 * \version   1st Version Aug. 1, 2005
 *
 ************************************************************/

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"

class ReadRecHitAlgorithm 
{
 public:
  
  ReadRecHitAlgorithm(const edm::ParameterSet& conf);
  ~ReadRecHitAlgorithm();
  

  /// Runs the algorithm
    void run(const SiStripRecHit2DCollection* input);
    void run(const SiStripMatchedRecHit2DCollection* input);

 private:

  edm::ParameterSet conf_;
};

#endif
