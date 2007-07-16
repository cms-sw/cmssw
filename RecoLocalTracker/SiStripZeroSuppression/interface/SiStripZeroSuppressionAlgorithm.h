#ifndef SiStripZeroSuppressionAlgorithm_h
#define SiStripZeroSuppressionAlgorithm_h

/** \class SiStripZeroSuppressionAlgorithm
 *
 * SiStripZeroSuppressionAlgorithm invokes specific strip Zero Suppression algorithms
 *
 * \author Domenico Giordano
 *
 *
 ************************************************************/

#include <string>

//edm
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//Data Formats
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
//ES Data
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"


#include <iostream> 
#include <memory>
#include <string>
#include <vector>
#include "CLHEP/Random/RandFlat.h"

class SiStripPedestalsService;
class SiStripNoiseService;
class SiStripPedestalsSubtractor;
class SiStripFedZeroSuppression;
class SiStripCommonModeNoiseSubtractor;


class SiStripZeroSuppressionAlgorithm 
{
 public:
  
  SiStripZeroSuppressionAlgorithm(const edm::ParameterSet& conf);
  ~SiStripZeroSuppressionAlgorithm();

  /// Runs the algorithm
  void run(std::string RawDigiType, const edm::DetSetVector<SiStripRawDigi>& input,
	   std::vector< edm::DetSet<SiStripDigi> >& output,edm::ESHandle<SiStripPedestals> &, edm::ESHandle<SiStripNoises> &);

 private:
  edm::ParameterSet conf_;

  SiStripFedZeroSuppression* SiStripZeroSuppressor_;
  std::string ZeroSuppressionMode_;
  bool validZeroSuppression_;  

  SiStripCommonModeNoiseSubtractor* SiStripCommonModeNoiseSubtractor_;
  std::string CMNSubtractionMode_;
  bool validCMNSubtraction_;  

  SiStripPedestalsSubtractor* SiStripPedestalsSubtractor_;
};
#endif
