#ifndef SiStrip1DLocalMeasurementConverterAlgorithm_h
#define SiStrip1DLocalMeasurementConverterAlgorithm_h

/** \class SiStrip1DLocalMeasurementConverterAlgorithm
 *
 * SiStrip1DLocalMeasurementConverterAlgorithm invokes specific strip clustering algorithms
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Aug. 1, 2005
 *
 ************************************************************/

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStrip1DLocalMeasurementCollection.h"

class SiStrip1DLocalMeasurementConverterAlgorithm 
{
 public:
  
  SiStrip1DLocalMeasurementConverterAlgorithm(const edm::ParameterSet& conf);
  ~SiStrip1DLocalMeasurementConverterAlgorithm();

  /// Runs the algorithm
  void run(const SiStripClusterCollection* input,
	   const edm::EventSetup& c,
	   SiStrip1DLocalMeasurementCollection &output);

 private:
  edm::ParameterSet conf_;

};

#endif
