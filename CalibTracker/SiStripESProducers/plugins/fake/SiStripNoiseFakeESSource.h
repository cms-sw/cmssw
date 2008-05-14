#ifndef CalibTracker_ESProducers_SiStripNoiseFakeESSource_H
#define CalibTracker_ESProducers_SiStripNoiseFakeESSource_H

//#include "FWCore/Framework/interface/ESProducer.h"
//#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CalibTracker/SiStripESProducers/interface/SiStripNoiseESSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "boost/cstdint.hpp"
#include <memory>


/** 
    @class SiStripNoiseFakeESSource
    @brief Fake source of SiStripNoises object.
    @author G. Bruno
*/
class SiStripNoiseFakeESSource : public SiStripNoiseESSource {

 public:

  SiStripNoiseFakeESSource( const edm::ParameterSet& );
  virtual ~SiStripNoiseFakeESSource() {;}
  
     
private:
  

  SiStripNoises* makeNoise();


private:

  //parameters for strip length proportional noise generation. not used if random mode is chosen
  double noiseStripLengthLinearSlope_;
  double noiseStripLengthLinearQuote_;
  double electronsPerADC_;

  bool printdebug_;
  edm::FileInPath fp_;

};


#endif // CalibTracker_ESProducers_SiStripNoiseFakeESSource_H

