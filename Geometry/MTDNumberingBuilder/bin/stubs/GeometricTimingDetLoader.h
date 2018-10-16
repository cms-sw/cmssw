#ifndef MTDNumberingBuilder_GeometricTimingDetLoader_h
#define MTDNumberingBuilder_GeometricTimingDetLoader_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class GeometricTimingDet;
class PGeometricTimingDet;

class GeometricTimingDetLoader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {

 public:
  explicit GeometricTimingDetLoader( const edm::ParameterSet& iConfig );
  ~GeometricTimingDetLoader() override;

  void beginJob() override {}
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  void endJob() override {}
  
 private:
  void putOne ( const GeometricTimingDet* gd, PGeometricTimingDet* pgd, int lev );
};

#endif
