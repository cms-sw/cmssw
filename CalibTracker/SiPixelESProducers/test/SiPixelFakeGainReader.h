#ifndef CalibTracker_SiPixelESProducers_test_SiPixelFakeGainReader
#define CalibTracker_SiPixelESProducers_test_SiPixelFakeGainReader
// -*- C++ -*-
//
// Package:    SiPixelFakeGainReader
// Class:      SiPixelFakeGainReader
//
/**\class SiPixelFakeGainReader SiPixelFakeGainReader.h SiPixelESProducers/test/SiPixelFakeGainReader.h

 Description: Test analyzer for fake pixel calibration

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo CHIOCHIA
//         Created:  Tue Oct 17 17:40:56 CEST 2006
//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationService.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"

namespace cms {
  class SiPixelFakeGainReader : public edm::EDAnalyzer {
  public:
    explicit SiPixelFakeGainReader(const edm::ParameterSet& iConfig);

    ~SiPixelFakeGainReader(){};
    virtual void beginJob();
    virtual void beginRun(const edm::Run&, const edm::EventSetup&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob();

  private:
    edm::ParameterSet conf_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomTokenBeginRun_;
    SiPixelGainCalibrationService SiPixelGainCalibrationService_;

    std::map<uint32_t, TH1F*> _TH1F_Pedestals_m;
    std::map<uint32_t, TH1F*> _TH1F_Gains_m;
    std::string filename_;
    TFile* fFile;
  };
}  // namespace cms
#endif
