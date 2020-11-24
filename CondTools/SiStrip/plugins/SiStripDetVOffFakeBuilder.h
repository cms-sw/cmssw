#ifndef SiStripDetVOff_H
#define SiStripDetVOff_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class TrackerGeometry;

class SiStripDetVOffFakeBuilder : public edm::EDAnalyzer {
public:
  explicit SiStripDetVOffFakeBuilder(const edm::ParameterSet& iConfig);

  ~SiStripDetVOffFakeBuilder() override;

  virtual void initialize(const edm::EventSetup&);

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  bool printdebug_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  std::vector<uint32_t> detids;
};
#endif
