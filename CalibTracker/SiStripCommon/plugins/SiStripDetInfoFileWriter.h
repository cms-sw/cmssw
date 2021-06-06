#ifndef CalibTracker_SiStripChannelGain_SiStripDetInfoFileWriter_h
#define CalibTracker_SiStripChannelGain_SiStripDetInfoFileWriter_h
// -*- C++ -*-
//
// Package:    SiStripDetInfoFileWriter
// Class:      SiStripDetInfoFileWriter
//
/**\class SiStripDetInfoFileWriter SiStripDetInfoFileWriter.cc CalibTracker/SiStripCommon/src/SiStripDetInfoFileWriter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  G. Bruno
//         Created:  Mon Nov 20 10:04:31 CET 2006
//
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
#include <iostream>
#include <fstream>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class SiStripDetInfoFileWriter : public edm::EDAnalyzer {
public:
  explicit SiStripDetInfoFileWriter(const edm::ParameterSet&);
  ~SiStripDetInfoFileWriter() override;

private:
  void beginRun(const edm::Run&, const edm::EventSetup& iSetup) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override{};

private:
  std::ofstream outputFile_;
  std::string filePath_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
};
#endif
