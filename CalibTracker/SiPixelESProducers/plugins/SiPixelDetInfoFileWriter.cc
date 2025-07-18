// -*- C++ -*-
// Package:    CalibTracker/SiPixelESProducers
// Class:      SiPixelDetInfoFileWriter
// Original Author:  V.Chiochia (adapted from the Strip version by G.Bruno)
//         Created:  Mon May 20 10:04:31 CET 2007

// system includes
#include <string>
#include <iostream>
#include <fstream>

// user includes
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class SiPixelDetInfoFileWriter : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit SiPixelDetInfoFileWriter(const edm::ParameterSet &);
  ~SiPixelDetInfoFileWriter() override;

private:
  void beginJob() override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override {}

private:
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomTokenBeginRun_;
  std::ofstream outputFile_;
  std::string filePath_;
};

using namespace cms;
using namespace std;

SiPixelDetInfoFileWriter::SiPixelDetInfoFileWriter(const edm::ParameterSet &iConfig) {
  edm::LogInfo("SiPixelDetInfoFileWriter::SiPixelDetInfoFileWriter");

  trackerGeomTokenBeginRun_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>();
  filePath_ = iConfig.getUntrackedParameter<std::string>("FilePath", std::string("SiPixelDetInfo.dat"));
}

SiPixelDetInfoFileWriter::~SiPixelDetInfoFileWriter() {
  edm::LogInfo("SiPixelDetInfoFileWriter::~SiPixelDetInfoFileWriter");
}

void SiPixelDetInfoFileWriter::beginRun(const edm::Run &run, const edm::EventSetup &iSetup) {
  outputFile_.open(filePath_.c_str());

  if (outputFile_.is_open()) {
    edm::ESHandle<TrackerGeometry> pDD = iSetup.getHandle(trackerGeomTokenBeginRun_);

    edm::LogInfo("SiPixelDetInfoFileWriter::beginJob - got geometry  ") << std::endl;
    edm::LogInfo("SiPixelDetInfoFileWriter") << " There are " << pDD->detUnits().size() << " detectors" << std::endl;

    int nPixelDets = 0;

    for (const auto &it : pDD->detUnits()) {
      const PixelGeomDetUnit *mit = dynamic_cast<PixelGeomDetUnit const *>(it);

      if (mit != nullptr) {
        nPixelDets++;
        const PixelTopology &topol = mit->specificTopology();
        // Get the module sizes.
        int nrows = topol.nrows();     // rows in x
        int ncols = topol.ncolumns();  // cols in y
        uint32_t detid = (mit->geographicalId()).rawId();

        outputFile_ << detid << " " << ncols << " " << nrows << "\n";
      }
    }
    outputFile_.close();
    edm::LogInfo("SiPixelDetInfoFileWriter::beginJob - Loop finished. ")
        << nPixelDets << " Pixel DetUnits found " << std::endl;
  }

  else {
    edm::LogError("SiPixelDetInfoFileWriter::beginJob - Unable to open file") << endl;
    return;
  }
}

void SiPixelDetInfoFileWriter::beginJob() {}

void SiPixelDetInfoFileWriter::analyze(const edm::Event &, const edm::EventSetup &) {}

DEFINE_FWK_MODULE(SiPixelDetInfoFileWriter);
