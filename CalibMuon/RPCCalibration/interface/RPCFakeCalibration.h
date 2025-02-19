#ifndef RPCCalibration_RPCFakeCalibration_H
#define RPCCalibration_RPCFakeCalibration_H

#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"
#include "CalibMuon/RPCCalibration/interface/RPCPerformanceESSource.h"
//#include "CalibMuon/RPCCalibration/interface/RPCClusterSizeESSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "boost/cstdint.hpp"
#include <memory>

class RPCCalibSetUp;

/** 
    @class RPCFakeCalibration
    @brief Fake source of RPCStripNoises object.
    @author R. Trentadue, B. Pavlov
*/

class RPCFakeCalibration : public RPCPerformanceESSource {

 public:

  RPCFakeCalibration( const edm::ParameterSet& );
  virtual ~RPCFakeCalibration() {;}
  
     
private:
  

  RPCStripNoises* makeNoise();

  RPCClusterSize* makeCls();

private:

  //  bool printdebug_;
  RPCCalibSetUp* theRPCCalibSetUp;

};


#endif // RPCCalibration_RPCFakeCalibration_H
