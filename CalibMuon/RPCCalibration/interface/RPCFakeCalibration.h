#ifndef RPCCalibration_RPCFakeCalibration_H
#define RPCCalibration_RPCFakeCalibration_H

#include "CalibMuon/RPCCalibration/interface/RPCPerformanceESSource.h"
#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"
//#include "CalibMuon/RPCCalibration/interface/RPCClusterSizeESSource.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class RPCCalibSetUp;

/**
    @class RPCFakeCalibration
    @brief Fake source of RPCStripNoises object.
    @author R. Trentadue, B. Pavlov
*/

class RPCFakeCalibration : public RPCPerformanceESSource {
public:
  RPCFakeCalibration(const edm::ParameterSet &);
  ~RPCFakeCalibration() override { ; }

private:
  RPCStripNoises *makeNoise() override;

  RPCClusterSize *makeCls();

private:
  //  bool printdebug_;
  RPCCalibSetUp *theRPCCalibSetUp;
};

#endif  // RPCCalibration_RPCFakeCalibration_H
