#ifndef SiPixelDynamicInefficiencyReader_H
#define SiPixelDynamicInefficiencyReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"
#include "CondFormats/DataRecord/interface/SiPixelDynamicInefficiencyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

//
//
// class decleration
//
class SiPixelDynamicInefficiencyReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelDynamicInefficiencyReader(const edm::ParameterSet&);
  ~SiPixelDynamicInefficiencyReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken;
  const edm::ESGetToken<SiPixelDynamicInefficiency, SiPixelDynamicInefficiencyRcd> dynIneffToken;

  bool printdebug_;
  double thePixelEfficiency[20];                     // Single pixel effciency
  double thePixelColEfficiency[20];                  // Column effciency
  double thePixelChipEfficiency[20];                 // ROC efficiency
  std::vector<double> theLadderEfficiency_BPix[20];  // Ladder efficiency
  std::vector<double> theModuleEfficiency_BPix[20];  // Module efficiency
  std::vector<double> thePUEfficiency[20];           // Instlumi dependent efficiency
  double theInnerEfficiency_FPix[20];                // Fpix inner module efficiency
  double theOuterEfficiency_FPix[20];                // Fpix outer module efficiency
};

#endif
