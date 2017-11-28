//
// Original Author:  Pawel Jurgielewicz
//         Created:  Tue, 21 Nov 2017 13:38:45 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"

#include "TGraph.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TH2Poly.h"
#include "TProfile2D.h"
#include "TColor.h"

#include <iostream>
#include <vector>
#include <map>
#include <string>

using namespace edm;
using namespace std;

class TrackerRemapper : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TrackerRemapper(const edm::ParameterSet&);
  ~TrackerRemapper() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  enum PixelLayerEnum {
    INVALID = 0,

    PXB_L1,
    PXB_L2,
    PXB_L3,
    PXB_L4,

    PXF_R1,
    PXF_R2
  };

  enum AnalyzeData {
    RECHITS = 1,
    DIGIS,
    CLUSTERS,
  };

  enum OpMode { MODE_ANALYZE = 0, MODE_REMAP = 1 };

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void ReadVertices(double& minx, double& maxx, double& miny, double& maxy);

  void PrepareStripNames();
  void PreparePixelNames();

  void BookBins(ESHandle<TrackerGeometry>& theTrackerGeometry, const TrackerTopology* tt);

  template <class T>
  void AnalyzeGeneric(const edm::Event& iEvent, const edm::EDGetTokenT<T>& src);
  void AnalyzeRechits(const edm::Event& iEvent, const edm::EDGetTokenT<reco::TrackCollection>& src);
  // void AnalyzeDigis(const edm::Event& iEvent);
  void AnalyzeClusters(const edm::Event& iEvent);

  void FillStripRemap();
  void FillPixelRemap(ESHandle<TrackerGeometry>& theTrackerGeometry, const TrackerTopology* tt);
  void FillBarrelRemap(TFile* rootFileHandle, ESHandle<TrackerGeometry>& theTrackerGeometry, const TrackerTopology* tt);
  void FillEndcapRemap(TFile* rootFileHandle, ESHandle<TrackerGeometry>& theTrackerGeometry, const TrackerTopology* tt);

  edm::Service<TFileService> fs;
  const edm::ParameterSet& iConfig;

  unsigned opMode;
  unsigned analyzeMode;

  std::map<long, TGraph*> bins;
  std::vector<unsigned> detIdVector;

  const TkDetMap* tkdetmap;

  map<unsigned, string> stripHistnameMap;
  map<unsigned, string> pixelHistnameMap;
  map<unsigned, string> analyzeModeNameMap;

  string stripRemapFile;
  string pixelRemapFile;

  string stripBaseDir, stripDesiredHistogram;
  string pixelBaseDir, pixelDesiredHistogramBarrel, pixelDesiredHistogramDisk;

  string runString;

  TH2Poly* trackerMap{nullptr};

  edm::EDGetTokenT<reco::TrackCollection> rechitSrcToken;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripDigi>> digiSrcToken;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> clusterSrcToken;
};

template <class T>
void TrackerRemapper::AnalyzeGeneric(const edm::Event& iEvent, const edm::EDGetTokenT<T>& src) {
  edm::Handle<T> input;
  iEvent.getByToken(src, input);

  if (!input.isValid()) {
    cout << "<GENERIC> not found... Aborting...\n";
    return;
  }

  typename T::const_iterator it;
  for (it = input->begin(); it != input->end(); ++it) {
    auto id = DetId(it->detId());
    trackerMap->Fill(TString::Format("%ld", (long)id.rawId()), it->size());
  }
}

template <>
void TrackerRemapper::AnalyzeGeneric(const edm::Event& iEvent, const edm::EDGetTokenT<reco::TrackCollection>& t) {
  AnalyzeRechits(iEvent, t);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerRemapper);
