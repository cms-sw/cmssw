#ifndef TrackerNumberingBuilder_GeometricDetLoader_h
#define TrackerNumberingBuilder_GeometricDetLoader_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class GeometricDet;
class PGeometricDet;

class GeometricDetLoader : public edm::EDAnalyzer {

 public:
  explicit GeometricDetLoader( const edm::ParameterSet& iConfig );
  ~GeometricDetLoader();
  virtual void beginJob( edm::EventSetup const& );
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};
  void putOne ( const GeometricDet* gd, PGeometricDet* pgd, int lev );
 private:

};

#endif
