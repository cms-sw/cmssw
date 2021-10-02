#ifndef CalibTracker_SiPixelDynamicInefficiencyDB_SiPixelDynamicInefficiencyDB_h
#define CalibTracker_SiPixelDynamicInefficiencyDB_SiPixelDynamicInefficiencyDB_h

#include <map>

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class SiPixelDynamicInefficiencyDB : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelDynamicInefficiencyDB(const edm::ParameterSet& conf);

  ~SiPixelDynamicInefficiencyDB() override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken_;
  edm::ParameterSet conf_;
  std::string recordName_;

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters thePixelGeomFactors_;
  Parameters theColGeomFactors_;
  Parameters theChipGeomFactors_;
  Parameters thePUEfficiency_;
  double theInstLumiScaleFactor_;
};

#endif
