#ifndef CalibTracker_SiPixelESProducers_SiPixelDetInfoFileWriter_h
#define CalibTracker_SiPixelESProducers_SiPixelDetInfoFileWriter_h
// -*- C++ -*-
//
// Package:    SiPixelDetInfoFileWriter
// Class:      SiPixelDetInfoFileWriter
//
/**\class SiPixelDetInfoFileWriter SiPixelDetInfoFileWriter.cc CalibTracker/SiPixelCommon/src/SiPixelDetInfoFileWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  Mon Nov 20 10:04:31 CET 2006
//
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include <string>
#include <iostream>
#include <fstream>

class SiPixelDetInfoFileWriter : public edm::EDAnalyzer {
public:
  explicit SiPixelDetInfoFileWriter(const edm::ParameterSet &);
  ~SiPixelDetInfoFileWriter() override;

private:
  void beginJob() override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomTokenBeginRun_;
  std::ofstream outputFile_;
  std::string filePath_;
};
#endif
