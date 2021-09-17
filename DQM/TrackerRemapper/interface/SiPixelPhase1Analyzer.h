#ifndef DQM_TRACKERREMAPPER_SIPIXELPHASE1ANALYZER_H
#define DQM_TRACKERREMAPPER_SIPIXELPHASE1ANALYZER_H

/**\class SiPixelPhase1Analyzer SiPixelPhase1Analyzer.cc EJTerm/SiPixelPhase1Analyzer/plugins/SiPixelPhase1Analyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Pawel Jurgielewicz
//         Created:  Tue, 21 Feb 2017 09:42:19 GMT
//
//

// system include files
#include <memory>

// #include <iostream>
#include <fstream>
#include <string>

#include <algorithm>

#include <vector>
#include <map>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DQM/TrackerRemapper/interface/mat4.h"
#include "DataFormats/GeometrySurface/interface/DiskSectorBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TH2.h"
#include "TProfile2D.h"
#include "TH2Poly.h"
#include "TGraph.h"

#define CODE_FORWARD(s, d, b) ((unsigned short)((b << 8) + (d << 4) + s))

//#define DEBUG_MODE

//
// class declaration
//

enum OperationMode { MODE_ANALYZE = 0, MODE_REMAP = 1 };

class SiPixelPhase1Analyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiPixelPhase1Analyzer(const edm::ParameterSet&);
  ~SiPixelPhase1Analyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void BookHistograms();

  void BookBarrelHistograms(TDirectory* currentDir, const std::string& currentHistoName);
  void BookForwardHistograms(TDirectory* currentDir, const std::string& currentHistoName);

  void BookBins(const TrackerGeometry& theTrackerGeometry, const TrackerTopology* tt);
  void BookBarrelBins(const TrackerGeometry& theTrackerGeometry, const TrackerTopology* tt);
  void BookForwardBins(const TrackerGeometry& theTrackerGeometry, const TrackerTopology* tt);

  void SaveDetectorVertices(const TrackerTopology* tt);

  void FillBins(edm::Handle<reco::TrackCollection>* tracks,
                const TrackerGeometry& theTrackerGeometry,
                const TrackerTopology* tt);

  void FillBarrelBinsAnalyze(const TrackerGeometry& theTrackerGeometry,
                             const TrackerTopology* tt,
                             unsigned rawId,
                             const GlobalPoint& globalPoint);
  void FillForwardBinsAnalyze(const TrackerGeometry& theTrackerGeometry,
                              const TrackerTopology* tt,
                              unsigned rawId,
                              const GlobalPoint& globalPoint);

  void FillBarrelBinsRemap(const TrackerGeometry& theTrackerGeometry, const TrackerTopology* tt);
  void FillForwardBinsRemap(const TrackerGeometry& theTrackerGeometry, const TrackerTopology* tt);

  // ----------member data ---------------------------
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  OperationMode opMode;

  edm::EDGetTokenT<reco::TrackCollection> tracksToken;

  std::string debugFileName;
  std::ofstream debugFile;

  edm::Service<TFileService> fs;

  bool firstEvent;

  std::map<uint32_t, TGraph*> bins, binsSummary;

  std::map<std::string, std::vector<TH2Poly*> > th2PolyBarrel;
  std::map<std::string, TH2Poly*> th2PolyBarrelSummary;

#ifdef DEBUG_MODE
  std::map<std::string, std::vector<TH2*> > th2PolyBarrelDebug;
#endif

  std::map<std::string, std::vector<TH2Poly*> > pxfTh2PolyForward;
  std::map<std::string, TH2Poly*> pxfTh2PolyForwardSummary;

#ifdef DEBUG_MODE
  std::map<std::string, std::vector<TH2*> > pxfTh2PolyForwardDebug;
#endif

  mat4 orthoProjectionMatrix;

  struct complementaryElements {
    mat4 mat[2];
    unsigned rawId[2];
  };
  // used to hold information about elements': ids & matrices which are of the same side, disk and barrel but different panel
  // to build trapezoidal ring elements
  std::map<unsigned short, complementaryElements> mapOfComplementaryElements;

  //Input root file handle;
  TFile* rootFileHandle;

  // read input histograms
  std::vector<unsigned> isBarrelSource;
  std::vector<std::string> analazedRootFileName;
  std::vector<std::string> pathToHistograms;
  std::vector<std::string> baseHistogramName;

  // temporal functionality
  void SaveDetectorData(bool isBarrel, unsigned rawId, int shell_hc, int layer_disk, int ladder_blade) {
    std::ofstream file("det.data", std::ofstream::out);

    file << isBarrel << "\t" << rawId << "\t" << shell_hc << "\t" << layer_disk << "\t" << ladder_blade << std::endl;
  }
};

#endif
