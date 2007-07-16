#ifndef CalibTracker_SiStripLorentzAngle_SiStripCalibLorentzAngle_h
#define CalibTracker_SiStripLorentzAngle_SiStripCalibLorentzAngle_h

#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"

class SiStripCalibLorentzAngle : public ConditionDBWriter<SiStripLorentzAngle>
{
 public:
  
  explicit SiStripCalibLorentzAngle(const edm::ParameterSet& conf);
  
  virtual ~SiStripCalibLorentzAngle();
  
  SiStripLorentzAngle* getNewObject();
  void algoBeginJob(const edm::EventSetup&);

 private:

  std::vector< std::pair<uint32_t, float> > detid_la;
  edm::ParameterSet conf_;
  //  double appliedVoltage_;
  //  double chargeMobility_;
  //  double temperature_;
  // double temperatureerror_;
  //double rhall_;
  //double holeBeta_;
  //double holeSaturationVelocity_;
};


#endif
