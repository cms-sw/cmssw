#ifndef CondToolsGeometry_PGeometricDetBuilder_h
#define CondToolsGeometry_PGeometricDetBuilder_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class GeometricDet;
class PGeometricDet;

class PGeometricDetBuilder : public edm::EDAnalyzer {

 public:
  explicit PGeometricDetBuilder( const edm::ParameterSet& iConfig );
  ~PGeometricDetBuilder();
  virtual void beginRun( const edm::Run&, edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};
  void putOne ( const GeometricDet* gd, PGeometricDet* pgd, int lev );
 private:
};

#endif
