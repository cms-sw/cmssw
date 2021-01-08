//
// Original Author:  Pawel Jurgielewicz
//         Created:  Tue, 21 Nov 2017 13:38:45 GMT
//
//

// system include files
#include <memory>
#include <fstream>
#include <utility>
#include <iostream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// root include files
#include "TGraph.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TH2Poly.h"
#include "TProfile2D.h"
#include "TColor.h"

using namespace edm;

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

  void readVertices(double& minx, double& maxx, double& miny, double& maxy);

  void prepareStripNames();
  void preparePixelNames();

  void bookBins();

  template <class T>
  void analyzeGeneric(const edm::Event& iEvent, const edm::EDGetTokenT<T>& src);
  void analyzeRechits(const edm::Event& iEvent, const edm::EDGetTokenT<reco::TrackCollection>& src);

  void fillStripRemap();
  void fillPixelRemap(const TrackerGeometry* theTrackerGeometry, const TrackerTopology* tt);
  void fillBarrelRemap(TFile* rootFileHandle, const TrackerGeometry* theTrackerGeometry, const TrackerTopology* tt);
  void fillEndcapRemap(TFile* rootFileHandle, const TrackerGeometry* theTrackerGeometry, const TrackerTopology* tt);

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_;

  edm::Service<TFileService> fs;
  const edm::ParameterSet& iConfig;

  int opMode;
  int analyzeMode;

  std::map<long, TGraph*> bins;
  std::vector<unsigned> detIdVector;

  const TkDetMap* tkdetmap;

  std::map<unsigned, std::string> stripHistnameMap;
  std::map<unsigned, std::string> pixelHistnameMap;
  std::map<unsigned, std::string> analyzeModeNameMap;

  std::string stripRemapFile;
  std::string pixelRemapFile;

  std::string stripBaseDir, stripDesiredHistogram;
  std::string pixelBaseDir, pixelDesiredHistogramBarrel, pixelDesiredHistogramDisk;

  std::string runString;

  TH2Poly* trackerMap{nullptr};

  edm::EDGetTokenT<reco::TrackCollection> rechitSrcToken;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripDigi>> digiSrcToken;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> clusterSrcToken;
};

template <class T>
//***************************************************************//
void TrackerRemapper::analyzeGeneric(const edm::Event& iEvent, const edm::EDGetTokenT<T>& src)
//***************************************************************//
{
  edm::Handle<T> input;
  iEvent.getByToken(src, input);

  if (!input.isValid()) {
    edm::LogError("TrackerRemapper") << "<GENERIC> not found... Aborting...\n";
    return;
  }

  typename T::const_iterator it;
  for (it = input->begin(); it != input->end(); ++it) {
    auto id = DetId(it->detId());
    trackerMap->Fill(TString::Format("%ld", (long)id.rawId()), it->size());
  }
}

template <>
//***************************************************************//
void TrackerRemapper::analyzeGeneric(const edm::Event& iEvent, const edm::EDGetTokenT<reco::TrackCollection>& t)
//***************************************************************//
{
  analyzeRechits(iEvent, t);
}

//***************************************************************//
TrackerRemapper::TrackerRemapper(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      topoToken_(esConsumes()),
      tkDetMapToken_(esConsumes()),
      iConfig(iConfig),
      opMode(iConfig.getParameter<int>("opMode")),
      analyzeMode(iConfig.getParameter<int>("analyzeMode")) {
  usesResource("TFileService");

  if (opMode == MODE_REMAP) {
    stripRemapFile = iConfig.getParameter<std::string>("stripRemapFile");
    stripDesiredHistogram = iConfig.getParameter<std::string>("stripHistogram");
    runString = iConfig.getParameter<std::string>("runString");

    pixelRemapFile = std::string("DQM_V0001_PixelPhase1_R000305516.root");

    stripBaseDir = std::string("DQMData/Run " + runString + "/SiStrip/Run summary/MechanicalView/");
    pixelBaseDir = std::string("DQMData/Run " + runString + "/PixelPhase1/Run summary/Phase1_MechanicalView/");

    pixelDesiredHistogramBarrel = std::string("adc_per_SignedModule_per_SignedLadder");
    pixelDesiredHistogramDisk = std::string("adc_per_PXDisk_per_SignedBladePanel");

    prepareStripNames();
    preparePixelNames();
  } else if (opMode == MODE_ANALYZE) {
    analyzeModeNameMap[RECHITS] = "# Rechits";
    analyzeModeNameMap[DIGIS] = "# Digis";
    analyzeModeNameMap[CLUSTERS] = "# Clusters";

    switch (analyzeMode) {
      case RECHITS:
        rechitSrcToken = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"));
        break;
      case DIGIS:
        digiSrcToken = consumes<edmNew::DetSetVector<SiStripDigi>>(iConfig.getParameter<edm::InputTag>("src"));
        break;
      case CLUSTERS:
        clusterSrcToken = consumes<edmNew::DetSetVector<SiStripCluster>>(iConfig.getParameter<edm::InputTag>("src"));
        break;
      default:
        edm::LogError("LogicError") << "Unrecognized analyze mode!" << std::endl;
    }
  } else {
    throw cms::Exception("TrackerRemapper") << "Unrecognized operations mode!" << std::endl;
  }

  // TColor::SetPalette(1);
}

//***************************************************************//
void TrackerRemapper::prepareStripNames()
//***************************************************************//
{
  stripHistnameMap[TkLayerMap::TkLayerEnum::TIB_L1] =
      stripBaseDir + "TIB/layer_1/" + stripDesiredHistogram + "_TIB_L1;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TIB_L2] =
      stripBaseDir + "TIB/layer_2/" + stripDesiredHistogram + "_TIB_L2;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TIB_L3] =
      stripBaseDir + "TIB/layer_3/" + stripDesiredHistogram + "_TIB_L3;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TIB_L4] =
      stripBaseDir + "TIB/layer_4/" + stripDesiredHistogram + "_TIB_L4;1";

  stripHistnameMap[TkLayerMap::TkLayerEnum::TIDM_D1] =
      stripBaseDir + "TID/MINUS/wheel_1/" + stripDesiredHistogram + "_TIDM_D1;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TIDM_D2] =
      stripBaseDir + "TID/MINUS/wheel_2/" + stripDesiredHistogram + "_TIDM_D2;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TIDM_D3] =
      stripBaseDir + "TID/MINUS/wheel_3/" + stripDesiredHistogram + "_TIDM_D3;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TIDP_D1] =
      stripBaseDir + "TID/PLUS/wheel_1/" + stripDesiredHistogram + "_TIDP_D1;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TIDP_D2] =
      stripBaseDir + "TID/PLUS/wheel_2/" + stripDesiredHistogram + "_TIDP_D2;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TIDP_D3] =
      stripBaseDir + "TID/PLUS/wheel_3/" + stripDesiredHistogram + "_TIDP_D3;1";

  stripHistnameMap[TkLayerMap::TkLayerEnum::TOB_L1] =
      stripBaseDir + "TOB/layer_1/" + stripDesiredHistogram + "_TOB_L1;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TOB_L2] =
      stripBaseDir + "TOB/layer_2/" + stripDesiredHistogram + "_TOB_L2;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TOB_L3] =
      stripBaseDir + "TOB/layer_3/" + stripDesiredHistogram + "_TOB_L3;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TOB_L4] =
      stripBaseDir + "TOB/layer_4/" + stripDesiredHistogram + "_TOB_L4;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TOB_L5] =
      stripBaseDir + "TOB/layer_5/" + stripDesiredHistogram + "_TOB_L5;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TOB_L6] =
      stripBaseDir + "TOB/layer_6/" + stripDesiredHistogram + "_TOB_L6;1";

  stripHistnameMap[TkLayerMap::TkLayerEnum::TECM_W1] =
      stripBaseDir + "TEC/MINUS/wheel_1/" + stripDesiredHistogram + "_TECM_W1;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECM_W2] =
      stripBaseDir + "TEC/MINUS/wheel_2/" + stripDesiredHistogram + "_TECM_W2;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECM_W3] =
      stripBaseDir + "TEC/MINUS/wheel_3/" + stripDesiredHistogram + "_TECM_W3;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECM_W4] =
      stripBaseDir + "TEC/MINUS/wheel_4/" + stripDesiredHistogram + "_TECM_W4;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECM_W5] =
      stripBaseDir + "TEC/MINUS/wheel_5/" + stripDesiredHistogram + "_TECM_W5;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECM_W6] =
      stripBaseDir + "TEC/MINUS/wheel_6/" + stripDesiredHistogram + "_TECM_W6;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECM_W7] =
      stripBaseDir + "TEC/MINUS/wheel_7/" + stripDesiredHistogram + "_TECM_W7;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECM_W8] =
      stripBaseDir + "TEC/MINUS/wheel_8/" + stripDesiredHistogram + "_TECM_W8;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECM_W9] =
      stripBaseDir + "TEC/MINUS/wheel_9/" + stripDesiredHistogram + "_TECM_W9;1";

  stripHistnameMap[TkLayerMap::TkLayerEnum::TECP_W1] =
      stripBaseDir + "TEC/PLUS/wheel_1/" + stripDesiredHistogram + "_TECP_W1;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECP_W2] =
      stripBaseDir + "TEC/PLUS/wheel_2/" + stripDesiredHistogram + "_TECP_W2;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECP_W3] =
      stripBaseDir + "TEC/PLUS/wheel_3/" + stripDesiredHistogram + "_TECP_W3;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECP_W4] =
      stripBaseDir + "TEC/PLUS/wheel_4/" + stripDesiredHistogram + "_TECP_W4;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECP_W5] =
      stripBaseDir + "TEC/PLUS/wheel_5/" + stripDesiredHistogram + "_TECP_W5;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECP_W6] =
      stripBaseDir + "TEC/PLUS/wheel_6/" + stripDesiredHistogram + "_TECP_W6;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECP_W7] =
      stripBaseDir + "TEC/PLUS/wheel_7/" + stripDesiredHistogram + "_TECP_W7;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECP_W8] =
      stripBaseDir + "TEC/PLUS/wheel_8/" + stripDesiredHistogram + "_TECP_W8;1";
  stripHistnameMap[TkLayerMap::TkLayerEnum::TECP_W9] =
      stripBaseDir + "TEC/PLUS/wheel_9/" + stripDesiredHistogram + "_TECP_W9;1";
}

//***************************************************************//
void TrackerRemapper::preparePixelNames()
//***************************************************************//
{
  pixelHistnameMap[PixelLayerEnum::PXB_L1] = pixelBaseDir + "PXBarrel/" + pixelDesiredHistogramBarrel + "_PXLayer_1;1";
  pixelHistnameMap[PixelLayerEnum::PXB_L2] = pixelBaseDir + "PXBarrel/" + pixelDesiredHistogramBarrel + "_PXLayer_2;1";
  pixelHistnameMap[PixelLayerEnum::PXB_L3] = pixelBaseDir + "PXBarrel/" + pixelDesiredHistogramBarrel + "_PXLayer_3;1";
  pixelHistnameMap[PixelLayerEnum::PXB_L4] = pixelBaseDir + "PXBarrel/" + pixelDesiredHistogramBarrel + "_PXLayer_4;1";

  pixelHistnameMap[PixelLayerEnum::PXF_R1] = pixelBaseDir + "PXForward/" + pixelDesiredHistogramDisk + "_PXRing_1;1";
  pixelHistnameMap[PixelLayerEnum::PXF_R2] = pixelBaseDir + "PXForward/" + pixelDesiredHistogramDisk + "_PXRing_2;1";
}

TrackerRemapper::~TrackerRemapper() {}

//***************************************************************//
void TrackerRemapper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
//***************************************************************//
{
  // get the ES products

  const TrackerGeometry* theTrackerGeometry = &iSetup.getData(geomToken_);
  const TrackerTopology* tt = &iSetup.getData(topoToken_);
  tkdetmap = &iSetup.getData(tkDetMapToken_);

  if (!trackerMap)
    bookBins();

  if (opMode == MODE_ANALYZE) {
    switch (analyzeMode) {
      case AnalyzeData::RECHITS:
        analyzeGeneric(iEvent, rechitSrcToken);
        break;
      case AnalyzeData::DIGIS:
        analyzeGeneric(iEvent, digiSrcToken);
        break;
      case AnalyzeData::CLUSTERS:
        analyzeGeneric(iEvent, clusterSrcToken);
        break;
      default:
        edm::LogError("LogicError") << "Unrecognized Analyze mode!" << std::endl;
        return;
    }
  } else if (opMode == MODE_REMAP) {
    fillStripRemap();
    fillPixelRemap(theTrackerGeometry, tt);
  }
}

//***************************************************************//
void TrackerRemapper::analyzeRechits(const edm::Event& iEvent, const edm::EDGetTokenT<reco::TrackCollection>& src)
//***************************************************************//
{
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(src, tracks);
  if (!tracks.isValid()) {
    LogInfo("Analyzer") << "reco::TrackCollection not found... Aborting...\n";
    return;
  }

  for (auto const& track : *tracks) {
    auto recHitsBegin = track.recHitsBegin();
    for (unsigned i = 0; i < track.recHitsSize(); ++i) {
      auto recHit = *(recHitsBegin + i);
      if (!recHit->isValid())
        continue;

      DetId id = recHit->geographicalId();
      unsigned subdetId = id.subdetId();

      //reject Pixel
      if (subdetId == PixelSubdetector::PixelBarrel || subdetId == PixelSubdetector::PixelEndcap)
        continue;

      // NRECHITS
      trackerMap->Fill(TString::Format("%ld", (long)id.rawId()), 1);
    }
  }
}

//***************************************************************//
void TrackerRemapper::bookBins()
//***************************************************************//
{
  // Read vertices from file
  double minx = 0xFFFFFF, maxx = -0xFFFFFF, miny = 0xFFFFFF, maxy = -0xFFFFFFF;
  readVertices(minx, maxx, miny, maxy);

  TObject* ghostObj = fs->make<TH2Poly>("ghost", "ghost", -1, 1, -1, 1);

  TDirectory* topDir = fs->getBareDirectory();
  topDir->cd();

  int margin = 50;
  std::string mapTitle;
  switch (opMode) {
    case MODE_ANALYZE:
      mapTitle = std::string(analyzeModeNameMap[analyzeMode] + " - " + runString);
      break;
    case MODE_REMAP:
      mapTitle = std::string(stripDesiredHistogram + " - " + runString);
      break;
  }

  trackerMap = new TH2Poly("Tracker Map", mapTitle.c_str(), minx - margin, maxx + margin, miny - margin, maxy + margin);
  trackerMap->SetFloat();
  trackerMap->SetOption("COLZ");
  trackerMap->SetStats(false);

  for (auto pair : bins) {
    trackerMap->AddBin(pair.second->Clone());
  }

  topDir->Add(trackerMap);

  ghostObj->Delete();  //not needed any more
}

//***************************************************************//
void TrackerRemapper::readVertices(double& minx, double& maxx, double& miny, double& maxy)
//***************************************************************//
{
  std::ifstream in;

  // TPolyline vertices stored at https://github.com/cms-data/DQM-SiStripMonitorClient
  in.open(edm::FileInPath("DQM/SiStripMonitorClient/data/Geometry/tracker_map_bare").fullPath().c_str());

  unsigned count = 0;

  if (!in.good()) {
    throw cms::Exception("TrackerRemapper") << "Error Reading File" << std::endl;
  }
  while (in.good()) {
    long detid = 0;
    double x[5], y[5];

    std::string line;
    std::getline(in, line);
    ++count;

    TString string(line);
    TObjArray* array = string.Tokenize(" ");
    int ix{0}, iy{0};
    bool isPixel{false};
    for (int i = 0; i < array->GetEntries(); ++i) {
      if (i == 0) {
        detid = static_cast<TObjString*>(array->At(i))->String().Atoll();

        // Drop Pixel Data
        DetId detId(detid);
        if (detId.subdetId() == PixelSubdetector::PixelBarrel || detId.subdetId() == PixelSubdetector::PixelEndcap) {
          isPixel = true;
          break;
        }
      } else {
        if (i % 2 == 0) {
          x[ix] = static_cast<TObjString*>(array->At(i))->String().Atof();

          if (x[ix] < minx)
            minx = x[ix];
          if (x[ix] > maxx)
            maxx = x[ix];

          ++ix;
        } else {
          y[iy] = static_cast<TObjString*>(array->At(i))->String().Atof();

          if (y[iy] < miny)
            miny = y[iy];
          if (y[iy] > maxy)
            maxy = y[iy];

          ++iy;
        }
      }
    }

    if (isPixel)
      continue;

    detIdVector.push_back(detid);
    bins[detid] = new TGraph(ix, x, y);
    bins[detid]->SetName(TString::Format("%ld", detid));
    bins[detid]->SetTitle(TString::Format("Module ID=%ld", detid));
  }
}

//***************************************************************//
void TrackerRemapper::fillStripRemap()
//***************************************************************//
{
  int nchX;
  int nchY;
  double lowX, highX;
  double lowY, highY;

  TFile* rootFileHandle = new TFile(stripRemapFile.c_str());

  for (int layer = TkLayerMap::TkLayerEnum::TIB_L1; layer <= TkLayerMap::TkLayerEnum::TECP_W9; ++layer) {
    tkdetmap->getComponents(layer, nchX, lowX, highX, nchY, lowY, highY);

    const TProfile2D* histHandle = (TProfile2D*)rootFileHandle->Get(stripHistnameMap[layer].c_str());

    if (!histHandle) {
      edm::LogError("TrackerRemapper") << "Could not find histogram:\n\t" << stripHistnameMap[layer] << std::endl;
      return;
    }

    for (unsigned binx = 1; binx <= (unsigned)nchX; ++binx) {
      for (unsigned biny = 1; biny <= (unsigned)nchY; ++biny) {
        long rawid = tkdetmap->getDetFromBin(layer, binx, biny);

        if (rawid)  //bin represents real module -> go to file
        {
          double val = histHandle->GetBinContent(binx, biny);

          // edm::LogInfo("TrackerRemapper") << rawid << " " << val << "\n";

          trackerMap->Fill(TString::Format("%ld", rawid), val);
        }
      }
    }
  }

  rootFileHandle->Close();
}

//***************************************************************//
void TrackerRemapper::fillPixelRemap(const TrackerGeometry* theTrackerGeometry, const TrackerTopology* tt)
//***************************************************************//
{
  TFile* rootFileHandle = new TFile(pixelRemapFile.c_str());

  if (!rootFileHandle) {
    edm::LogError("TrackerRemapper") << "Could not find file:\n\t" << pixelRemapFile << std::endl;
    return;
  }
  fillBarrelRemap(rootFileHandle, theTrackerGeometry, tt);
  fillEndcapRemap(rootFileHandle, theTrackerGeometry, tt);

  rootFileHandle->Close();
}

//***************************************************************//
void TrackerRemapper::fillBarrelRemap(TFile* rootFileHandle,
                                      const TrackerGeometry* theTrackerGeometry,
                                      const TrackerTopology* tt)
//***************************************************************//
{
  TrackingGeometry::DetContainer pxb = theTrackerGeometry->detsPXB();

  for (auto& i : pxb) {
    const GeomDet* det = i;

    PXBDetId id = det->geographicalId();
    long rawid = id.rawId();

    int module = tt->pxbModule(id);
    int layer = tt->pxbLayer(id);

    int signedOnlineModule = module - 4;
    if (signedOnlineModule <= 0)
      --signedOnlineModule;

    PixelBarrelName pixelBarrelName = PixelBarrelName(id, tt, true);
    int onlineShell = pixelBarrelName.shell();

    int signedOnlineLadder = ((onlineShell & 1) ? -pixelBarrelName.ladderName() : pixelBarrelName.ladderName());

    const TProfile2D* histHandle = (TProfile2D*)rootFileHandle->Get(pixelHistnameMap[layer].c_str());

    unsigned nx = histHandle->GetNbinsX();
    unsigned ny = histHandle->GetNbinsY();

    unsigned binX = signedOnlineModule + ((nx + 1) >> 1);
    unsigned binY = (signedOnlineLadder) + ((ny + 1) >> 1);

    double val = histHandle->GetBinContent(binX, binY);

    trackerMap->Fill(TString::Format("%ld", rawid), val);
  }
}

//***************************************************************//
void TrackerRemapper::fillEndcapRemap(TFile* rootFileHandle,
                                      const TrackerGeometry* theTrackerGeometry,
                                      const TrackerTopology* tt)
//***************************************************************//
{
  TrackingGeometry::DetContainer pxf = theTrackerGeometry->detsPXF();

  for (auto& i : pxf) {
    const GeomDet* det = i;

    PXFDetId id = det->geographicalId();

    int side = tt->side(id);
    int disk = tt->layer(id);

    long rawid = id.rawId();

    PixelEndcapName pixelEndcapName = PixelEndcapName(PXFDetId(rawid), tt, true);

    unsigned layer = pixelEndcapName.ringName() - 1 + PixelLayerEnum::PXF_R1;
    const TProfile2D* histHandle = (TProfile2D*)rootFileHandle->Get(pixelHistnameMap[layer].c_str());

    // ---- REMAP (Online -> Offline)
    unsigned nx = histHandle->GetNbinsX();
    unsigned ny = histHandle->GetNbinsY();

    int onlineBlade = pixelEndcapName.bladeName();
    bool isInnerOnlineBlade = !(pixelEndcapName.halfCylinder() & 1);  // inner -> blade > 0 (?)

    int signedOnlineBlade = (isInnerOnlineBlade) ? onlineBlade : -onlineBlade;
    int signedDisk = (side == 2) ? disk : -disk;
    int pannel = pixelEndcapName.pannelName() - 1;

    unsigned binX = signedDisk + ((nx + 1) >> 1);
    unsigned binY = (signedOnlineBlade * 2) + (ny >> 1);

    double val = histHandle->GetBinContent(binX, binY + pannel);

    trackerMap->Fill(TString::Format("%ld", rawid), val);
  }
}

void TrackerRemapper::beginJob() {}

void TrackerRemapper::endJob() {}

//***************************************************************//
void TrackerRemapper::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
//***************************************************************//
{
  edm::ParameterSetDescription desc;
  desc.setComment(
      "Creates TH2Poly Strip Tracker maps by either analyzing the event or remapping exising DQM historams");
  desc.add<edm::InputTag>("src", edm::InputTag("generalTracks"));
  desc.ifValue(edm::ParameterDescription<int>("opMode", 0, true),
               0 >> edm::EmptyGroupDescription() or
                   1 >> (edm::ParameterDescription<std::string>("stripRemapFile", "", true) and
                         edm::ParameterDescription<std::string>("stripHistogram", "", true) and
                         edm::ParameterDescription<std::string>("runString", "", true)))
      ->setComment("0 for Analyze, 1 for Remap");

  desc.ifValue(edm::ParameterDescription<int>("analyzeMode", 1, true), edm::allowedValues<int>(1, 2, 3))
      ->setComment("1=Rechits, 2=Digis, 3=Clusters");

  //desc.add<unsigned int>("analyzeMode", 1)->setComment("1=Rechits, 2=Digis, 3=Clusters");
  //desc.add<unsigned int>("opMode", 0)->setComment("0 for Analyze, 1 for Remap");
  //desc.addOptional<std::string>("stripRemapFile","" )->setComment("file name to analyze, will come from the config file");
  //desc.addOptional<std::string>("stripHistogram","TkHMap_NumberValidHits" )->setComment("histogram to use to remap");
  //desc.addOptional<std::string>("runString", "")->setComment("run number, will come form config file");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerRemapper);
