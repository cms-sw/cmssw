#ifndef CalibTracker_RPCCalibration_RPCFakeCalibration_H
#define CalibTracker_RPCCalibration_RPCFakeCalibration_H

#include "CalibMuon/RPCCalibration/interface/RPCPerformanceESSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "boost/cstdint.hpp"
#include <memory>

class RPCSimSetUp;

/** 
    @class RPCFakeCalibration
    @brief Fake source of RPCStripNoises object.
    @author R. Trentadue
*/

class RPCFakeCalibration : public RPCPerformanceESSource {

 public:

  RPCFakeCalibration( const edm::ParameterSet& );
  virtual ~RPCFakeCalibration() {;}
  
     
private:
  

  RPCStripNoises* makeNoise();


private:

  //  bool printdebug_;
  RPCSimSetUp* theRPCSimSetUp;

};


#endif // CalibTracker_RPCCalibration_RPCFakeCalibration_H
