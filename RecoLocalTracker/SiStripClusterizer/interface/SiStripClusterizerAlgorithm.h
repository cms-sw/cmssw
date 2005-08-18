#ifndef SiStripClusterizerAlgorithm_h
#define SiStripClusterizerAlgorithm_h

/** \class SiStripClusterizerAlgorithm
 *
 * SiStripClusterizerAlgorithm invokes specific strip clustering algorithms
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Aug. 1, 2005
 *
 ************************************************************/

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

class ThreeThresholdStripClusterizer;

class SiStripClusterizerAlgorithm 
{
 public:
  
  SiStripClusterizerAlgorithm(const edm::ParameterSet& conf);
  ~SiStripClusterizerAlgorithm();

  /// Runs the algorithm
  void run(const StripDigiCollection* input,
	   SiStripClusterCollection &output);

 private:
  edm::ParameterSet conf_;
  ThreeThresholdStripClusterizer *threeThreshold_;
  std::string clusterMode_;
  bool validClusterizer_;

};

#endif
