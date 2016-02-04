#ifndef CondToolsGeometry_XMLGeometryBuilder_h
#define CondToolsGeometry_XMLGeometryBuilder_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>


class XMLGeometryBuilder : public edm::EDAnalyzer {

 public:
  explicit XMLGeometryBuilder( const edm::ParameterSet& iConfig );
  ~XMLGeometryBuilder();
  virtual void beginJob();
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob() {};
 private:
  std::string fname;
  bool zip;
};

#endif
