#ifndef CalibTracker_SiStripLorentzAngleDB_SiStripLorentzAngleDB_h
#define CalibTracker_SiStripLorentzAngleDB_SiStripLorentzAngleDB_h

#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"


class SiStripLorentzAngleDB : public edm::EDAnalyzer
{
 public:
  
  explicit SiStripLorentzAngleDB(const edm::ParameterSet& conf);
  
  virtual ~SiStripLorentzAngleDB();
  
  virtual void beginJob(const edm::EventSetup& c);
  
  virtual void endJob(); 
  
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  
 private:

  std::vector< std::pair<uint32_t, float> > detid_la;
  edm::ParameterSet conf_;
  double appliedVoltage_;
  double chargeMobility_;
  double temperature_;
  double rhall_;
  double holeBeta_;
  double holeSaturationVelocity_;
  
};


#endif
