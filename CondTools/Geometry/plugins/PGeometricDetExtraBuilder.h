#ifndef CondToolsGeometry_PGeometricDetExtraBuilder_h
#define CondToolsGeometry_PGeometricDetExtraBuilder_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class GeometricDetExtra;
class PGeometricDetExtra;

class PGeometricDetExtraBuilder : public edm::EDAnalyzer {

 public:
  explicit PGeometricDetExtraBuilder( const edm::ParameterSet& iConfig );
  ~PGeometricDetExtraBuilder();
  virtual void beginRun( const edm::Run&, edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};
  void putOne ( const GeometricDetExtra& gde, PGeometricDetExtra* pgde );
 private:
};

#endif
