#ifndef TrackerNumberingBuilder_GeometricDetLoader_h
#define TrackerNumberingBuilder_GeometricDetLoader_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class GeometricDet;
class PGeometricDet;

class GeometricDetLoader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {

 public:
  explicit GeometricDetLoader( const edm::ParameterSet& iConfig );
  ~GeometricDetLoader();

  void beginJob() override {}
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  void endJob() override {}
  
 private:
  void putOne ( const GeometricDet* gd, PGeometricDet* pgd, int lev );
};

#endif
