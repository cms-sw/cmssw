#ifndef CondToolsGeometry_GeometryTester_h
#define CondToolsGeometry_GeometryTester_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class GeometricDet;
class PGeometricDet;

class GeometryTester : public edm::EDAnalyzer {

 public:
  explicit GeometryTester( const edm::ParameterSet& iConfig );
  ~GeometryTester();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

 private:

  bool xmltest, tktest, ecaltest, hcaltest, calotowertest, castortest, zdctest, csctest, dttest, rpctest;
  std::string geomLabel_;
};

#endif
