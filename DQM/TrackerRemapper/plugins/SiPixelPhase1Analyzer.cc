// -*- C++ -*-
//
// Package:    DQM/TrackerRemapper
// Class:      SiPixelPhase1Analyzer
//

#include "DQM/TrackerRemapper/interface/SiPixelPhase1Analyzer.h"

using namespace std;
using namespace edm;

SiPixelPhase1Analyzer::SiPixelPhase1Analyzer(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      topoToken_(esConsumes()),
      opMode(static_cast<OperationMode>(iConfig.getUntrackedParameter<unsigned int>("opMode"))),
      debugFileName(iConfig.getUntrackedParameter<string>("debugFileName")),
      firstEvent(true),
      rootFileHandle(nullptr),
      isBarrelSource(iConfig.getUntrackedParameter<vector<unsigned>>("isBarrelSource")),
      analazedRootFileName(iConfig.getUntrackedParameter<vector<string>>("remapRootFileName")),
      pathToHistograms(iConfig.getUntrackedParameter<vector<string>>("pathToHistograms")),
      baseHistogramName(iConfig.getUntrackedParameter<vector<string>>("baseHistogramName")) {
#ifdef DEBUG_MODE
  debugFile = std::ofstream(debugFileName.c_str(), std::ofstream::out);
#endif
  usesResource("TFileService");

  orthoProjectionMatrix.BuildOrthographicMatrix(1.0f, -1.0f, 1.0f, -1.0f, -10.0f, 10.0f);

  switch (opMode) {
    case MODE_ANALYZE:

      tracksToken = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("src"));

      analazedRootFileName.clear();

      pathToHistograms.clear();
      pathToHistograms.push_back("RecHits/");

      baseHistogramName.clear();
      baseHistogramName.push_back("RecHits");

      break;
    case MODE_REMAP:
      break;
    default:
      break;
  }
}

SiPixelPhase1Analyzer::~SiPixelPhase1Analyzer() {
  for (auto& i : bins) {
    delete i.second;
  }

  for (auto& i : binsSummary) {
    delete i.second;
  }

#ifdef DEBUG_MODE
  debugFile.close();
#endif
}

// ------------ method called for each event  ------------
void SiPixelPhase1Analyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& theTrackerGeometry = iSetup.getData(geomToken_);
  const auto& tt = &iSetup.getData(topoToken_);

  if (firstEvent) {
    /////////////////////////////////
    BookHistograms();
    /////////////////////////////////
    BookBins(theTrackerGeometry, tt);
    /////////////////////////////////
    if (opMode == MODE_REMAP) {
      FillBins(nullptr, theTrackerGeometry, tt);
    }
    /////////////////////////////////
    firstEvent = false;
  }
  if (opMode == MODE_ANALYZE) {
    edm::Handle<reco::TrackCollection> tracks;
    iEvent.getByToken(tracksToken, tracks);
    if (!tracks.isValid()) {
      LogInfo("Analyzer") << "reco::TrackCollection not found... Aborting...\n";
      return;
    }
    FillBins(&tracks, theTrackerGeometry, tt);
  }
  // debugFile << "SiPixelPhase1Analyzer::analyze() - Event " << iEvent.run() << "/" << iEvent.id().event() << endl;
}

void SiPixelPhase1Analyzer::BookHistograms() {
  // ghost object <initializes> FileService (without it, it crashes and creation of directories would not be possible)
  TObject* ghostObj = fs->make<TH2Poly>("ghost", "ghost", -1, 1, -1, 1);

  TDirectory* topDir = fs->getBareDirectory();
  topDir->cd();

#ifdef DEBUG_MODE
  debugFile << "Full path: " << fs->fullPath() << endl << endl;
#endif
  string histName;
  for (unsigned j = 0; j < baseHistogramName.size(); ++j) {
    string currentHistoName = baseHistogramName[j];

    TDirectory* currentDir = topDir->mkdir(currentHistoName.c_str());
    currentDir->cd();

    if (opMode == MODE_REMAP) {
      if (isBarrelSource[j]) {
        BookBarrelHistograms(currentDir, currentHistoName);
      } else {
        BookForwardHistograms(currentDir, currentHistoName);
      }
    } else {
      BookBarrelHistograms(currentDir, currentHistoName);
      BookForwardHistograms(currentDir, currentHistoName);
    }

    topDir->cd();
  }

  ghostObj->Delete();  //not needed any more
}

void SiPixelPhase1Analyzer::BookBarrelHistograms(TDirectory* currentDir, const string& currentHistoName) {
  string histName;
  TH2Poly* th2p;

#ifdef DEBUG_MODE
  TH2* th2;
#endif

  for (unsigned i = 0; i < 4; ++i) {
    histName = "barrel_layer_";

    th2p = new TH2Poly((histName + std::to_string(i + 1)).c_str(), "PXBMap", -15.0, 15.0, 0.0, 5.0);

    th2p->SetFloat();

    th2p->GetXaxis()->SetTitle("z [cm]");
    th2p->GetYaxis()->SetTitle("ladder");

#ifdef DEBUG_MODE
    th2p->SetOption("COLZ 0 TEXT");
#else
    th2p->SetOption("COLZ L");
#endif

    currentDir->Add(th2p);
    th2PolyBarrel[currentHistoName].push_back(th2p);

#ifdef DEBUG_MODE
    if (opMode == MODE_ANALYZE) {
      th2 = new TH2I((histName + std::to_string(i + 1) + "_DEBUG").c_str(),
                     "position",
                     3000,
                     -30.0f,
                     30.0f,
                     1000,
                     -4.5f * (i + 1),
                     4.5f * (i + 1));

      th2->GetXaxis()->SetTitle("z [cm]");
      th2->GetYaxis()->SetTitle("-x [?]");

      th2->SetOption("COLZ 0 TEXT");

      currentDir->Add(th2);
      th2PolyBarrelDebug[currentHistoName].push_back(th2);
    }
#endif
  }

  th2p = new TH2Poly("barrel_summary", "PXBMap", -5.0, 5.0, 0.0, 5.0);
  th2p->SetFloat();

  th2p->GetXaxis()->SetTitle("");
  th2p->GetYaxis()->SetTitle("~ladder");

  th2p->SetOption("COLZ L");

  currentDir->Add(th2p);
  th2PolyBarrelSummary[currentHistoName] = th2p;
}

void SiPixelPhase1Analyzer::BookForwardHistograms(TDirectory* currentDir, const string& currentHistoName) {
  string histName;
  TH2Poly* th2p;
#ifdef DEBUG_MODE
  TH2* th2;
#endif

  for (unsigned side = 1; side <= 2; ++side) {
    for (unsigned disk = 1; disk <= 3; ++disk) {
      histName = "forward_disk_";

      th2p = new TH2Poly((histName + std::to_string((side == 1 ? -(int(disk)) : (int)disk))).c_str(),
                         "PXFMap",
                         -15.0,
                         15.0,
                         -15.0,
                         15.0);

      th2p->SetFloat();

      th2p->GetXaxis()->SetTitle("x [cm]");
      th2p->GetYaxis()->SetTitle("y [cm]");

#ifdef DEBUG_MODE
      th2p->SetOption("COLZ 0 TEXT");
#else
      th2p->SetOption("COLZ L");
#endif
      currentDir->Add(th2p);
      pxfTh2PolyForward[currentHistoName].push_back(th2p);

#ifdef DEBUG_MODE
      if (opMode == MODE_ANALYZE) {
        th2 = new TH2I((histName + std::to_string((side == 1 ? -(int(disk)) : (int)disk)) + "_DEBUG").c_str(),
                       "position",
                       1000,
                       -15.0f,
                       15.0f,
                       1000,
                       -15.0f,
                       15.0f);

        th2->GetXaxis()->SetTitle("x [cm]");
        th2->GetYaxis()->SetTitle("y [cm]");

        th2->SetOption("COLZ 0 TEXT");

        currentDir->Add(th2);
        pxfTh2PolyForwardDebug[currentHistoName].push_back(th2);
      }
#endif
    }
  }

  th2p = new TH2Poly("forward_summary", "PXFMap", -40.0, 50.0, -20.0, 90.0);
  th2p->SetFloat();

  th2p->GetXaxis()->SetTitle("");
  th2p->GetYaxis()->SetTitle("");

#ifdef DEBUG_MODE
  th2p->SetOption("COLZ 0 TEXT");
#else
  th2p->SetOption("COLZ L");
#endif

  currentDir->Add(th2p);
  pxfTh2PolyForwardSummary[currentHistoName] = th2p;
}

void SiPixelPhase1Analyzer::BookBins(const TrackerGeometry& theTrackerGeometry, const TrackerTopology* tt) {
  BookBarrelBins(theTrackerGeometry, tt);
  BookForwardBins(theTrackerGeometry, tt);

#ifdef DEBUG_MODE
  SaveDetectorVertices(tt);
#endif
}

void SiPixelPhase1Analyzer::BookBarrelBins(const TrackerGeometry& theTrackerGeometry, const TrackerTopology* tt) {
  TrackingGeometry::DetContainer pxb = theTrackerGeometry.detsPXB();
#ifdef DEBUG_MODE
  debugFile << "There are " << pxb.size() << " detector elements in the PXB." << endl;
#endif
  for (auto& i : pxb) {
    const GeomDet* det = i;

    PXBDetId id = det->geographicalId();

    Local2DPoint origin;
    GlobalPoint p = det->surface().toGlobal(origin);

    int layer = tt->pxbLayer(id);
    int ladder = tt->pxbLadder(id);

#ifdef DEBUG_MODE
    int module = tt->pxbModule(id);
    PixelBarrelName pixelBarrelName(id, tt, true);
    SaveDetectorData(
        true, id.rawId(), pixelBarrelName.shell(), pixelBarrelName.layerName(), pixelBarrelName.ladderName());
#endif

#ifdef DEBUG_MODE
    float r = sqrt(p.x() * p.x() + p.y() * p.y());

    debugFile << "Layer: " << layer << "\tLadder: " << ladder << "\tModule: " << module << "\t(x, y, z, r2): (" << p.x()
              << ", " << p.y() << ", " << p.z() << ", " << r << ")" << endl;
#endif

    const Bounds& b = (det->surface().bounds());
    float bl = b.length();

#ifdef DEBUG_MODE
    float bw = b.width();
    float bt = b.thickness();

    debugFile << "Length: " << bl << "\tWidth: " << bw << "\tThickness: " << bt << endl;
#endif

    float vertX[] = {p.z() - bl * 0.5f, p.z() + bl * 0.5f, p.z() + bl * 0.5f, p.z() - bl * 0.5f, p.z() - bl * 0.5f};
    float vertY[] = {(ladder - 1.0f), (ladder - 1.0f), (float)ladder, (float)ladder, (ladder - 1.0f)};

    bins[id.rawId()] = new TGraph(5, vertX, vertY);
    bins[id.rawId()]->SetName(TString::Format("%u", id.rawId()));

    // Summary plot
    for (unsigned k = 0; k < 5; ++k) {
      vertX[k] += ((layer == 2 || layer == 3) ? 0.0f : -60.0f);
      vertY[k] += ((layer > 2) ? 30.0f : 0.0f);
    }

    binsSummary[id.rawId()] = new TGraph(5, vertX, vertY);
    binsSummary[id.rawId()]->SetName(TString::Format("%u", id.rawId()));

    for (unsigned nameNum = 0; nameNum < baseHistogramName.size(); ++nameNum) {
      if (isBarrelSource[nameNum] || opMode == MODE_ANALYZE) {
        const string& strName = baseHistogramName[nameNum];
        th2PolyBarrel[strName][layer - 1]->AddBin(bins[id.rawId()]->Clone());
        th2PolyBarrelSummary[strName]->AddBin(binsSummary[id.rawId()]->Clone());
      }
    }
  }
}

void SiPixelPhase1Analyzer::BookForwardBins(const TrackerGeometry& theTrackerGeometry, const TrackerTopology* tt) {
  TrackingGeometry::DetContainer pxf = theTrackerGeometry.detsPXF();
#ifdef DEBUG_MODE
  debugFile << "There are " << pxf.size() << " detector elements in the PXF." << endl;
#endif
  bool firstForwardElem = true;

  float elemWidth = 1.0f;
  float elemLength = 1.0f;

  // FIRST PASS -> MAKE MAP OF CORRESPONDING ELEMENTS (BLADES ON BOTH PANELS)
  for (auto& i : pxf) {
    const GeomDet* det = i;

    PXFDetId id = det->geographicalId();

    Local2DPoint origin;
    GlobalPoint p = det->surface().toGlobal(origin);

    int panel = tt->pxfPanel(id);
    int side = tt->side(id);   //tt->pxfSide(id);
    int disk = tt->layer(id);  //tt->pxfDisk(id);
    int blade = tt->pxfBlade(id);

#ifdef DEBUG_MODE
    int module = tt->module(id);  //tt->pxfModule(id);
    PixelEndcapName pixelEndcapName(id, tt, true);
    SaveDetectorData(
        false, id.rawId(), pixelEndcapName.halfCylinder(), pixelEndcapName.diskName(), pixelEndcapName.bladeName());
#endif

#ifdef DEBUG_MODE
    float r = sqrt(p.x() * p.x() + p.y() * p.y());

    debugFile << "Panel: " << panel << "\tSide: " << side << "\tDisk: " << disk << "\tBlade: " << blade
              << "\tModule: " << module << "\t(x, y, z, r): (" << p.x() << ", " << p.y() << ", " << p.z() << ", " << r
              << ")" << endl;
#endif
    if (firstForwardElem) {
      const Bounds& b = det->surface().bounds();  //const RectangularPlaneBounds& b

      elemLength = b.length();
      elemWidth = b.width();

      firstForwardElem = false;
    }

    const auto& rot = det->rotation();

    mat4 transMat(
        rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz(), p.x(), p.y(), p.z());

    mapOfComplementaryElements[CODE_FORWARD(side, disk, blade)].mat[panel - 1] = transMat;
    mapOfComplementaryElements[CODE_FORWARD(side, disk, blade)].rawId[panel - 1] = id.rawId();
  }

  // SECOND PASS -> USE INFORMATION FROM MAP TO MAKE GEOMETRIC OBJECTS (BINS)
  for (auto& i : mapOfComplementaryElements) {
    // decode side&disk from the map key
    unsigned side = i.first & 0xF;
    unsigned disk = (i.first & 0xF0) >> 4;
    // unsigned blade = (i.first & 0xFF00) >> 8;

    unsigned mapIdx = disk + (side - 1) * 3 - 1;

    // normal vectors of elements point to the (almost) opposite direction, so correction is needed before interploation (probably not 100% correct but fast)
    i.second.mat[1].data[0] = -i.second.mat[1].data[0];
    i.second.mat[1].data[1] = -i.second.mat[1].data[1];
    i.second.mat[1].data[2] = -i.second.mat[1].data[2];

    i.second.mat[1].data[6] = -i.second.mat[1].data[6];
    i.second.mat[1].data[7] = -i.second.mat[1].data[7];
    i.second.mat[1].data[8] = -i.second.mat[1].data[8];

    mat4 meanTransform = (i.second.mat[0] + i.second.mat[1]) * 0.5f;
    // mat4 meanTransform = i.second.mat[0];

    static const float baseVertX[4] = {-elemWidth * 0.8f, -elemWidth * 0.5f, elemWidth * 0.8f, elemWidth * 0.5f};
    static const float baseVertY[4] = {
        elemLength * 0.38f, -elemLength * 0.38f, elemLength * 0.38f, -elemLength * 0.38f};

    float vertXPanel[2][4], vertYPanel[2][4];
    float vertIn[3], vertOut[3];

    /*				
		(1)  __________________ (3)
		     \		       /
		      \		      /
		       \             /
		        \	    /
		      (2)\_________/(4)
		     
		     - division line: (2) - (3)
		     
		     - panel 1: triangle of lower area (2, 3, 4)
		     - panel 2: triangle of bigger area (1, 2, 3)
    */

    // obtain transformed vertices
    for (unsigned j = 0; j < 4; ++j) {
      vertIn[0] = baseVertX[j];
      vertIn[1] = baseVertY[j];
      vertIn[2] = 0.0f;

      meanTransform.MulVec(vertIn, vertOut);
      std::swap(vertIn, vertOut);
      orthoProjectionMatrix.MulVec(vertIn, vertOut);

      // vertical flip
      vertOut[0] = -vertOut[0];  // so that inner elements have positive x-coordinate

      if (j > 0) {
        vertXPanel[0][j - 1] = vertOut[0];
        vertYPanel[0][j - 1] = vertOut[1];  // for panel 2
      }
      if (j < 3) {
        vertXPanel[1][j] = vertOut[0];
        vertYPanel[1][j] = vertOut[1];  // for panel 1
      }
    }

    for (unsigned j = 0; j < 2; ++j) {
      vertXPanel[j][3] = vertXPanel[j][0];
      vertYPanel[j][3] = vertYPanel[j][0];

      bins[i.second.rawId[j]] = new TGraph(4, vertXPanel[j], vertYPanel[j]);
      bins[i.second.rawId[j]]->SetName(TString::Format("%u", i.second.rawId[j]));

      // for (auto strName: baseHistogramName)
      // {
      // pxfTh2PolyForward[strName][mapIdx]->AddBin(bins[i.second.rawId[j]]->Clone());
      // }

      // Summary plot
      for (unsigned k = 0; k < 4; ++k) {
        vertXPanel[j][k] += (float(side) - 1.5f) * 40.0f;
        vertYPanel[j][k] += (disk - 1) * 35.0f;
      }

      binsSummary[i.second.rawId[j]] = new TGraph(4, vertXPanel[j], vertYPanel[j]);
      binsSummary[i.second.rawId[j]]->SetName(TString::Format("%u", i.second.rawId[j]));

      for (unsigned nameNum = 0; nameNum < baseHistogramName.size(); ++nameNum) {
        if (!isBarrelSource[nameNum] || opMode == MODE_ANALYZE) {
          const string& strName = baseHistogramName[nameNum];
          pxfTh2PolyForward[strName][mapIdx]->AddBin(bins[i.second.rawId[j]]->Clone());
          pxfTh2PolyForwardSummary[strName]->AddBin(binsSummary[i.second.rawId[j]]->Clone());
        }
      }
    }
  }
}

void SiPixelPhase1Analyzer::SaveDetectorVertices(const TrackerTopology* tt) {
  vector<std::ofstream*> verticesFiles[2];
  for (unsigned i = 0; i < 4; ++i) {
    std::ofstream* f = new std::ofstream(("vertices_barrel_" + std::to_string(i + 1)).c_str(), std::ofstream::out);

    verticesFiles[0].push_back(f);
  }

  for (unsigned side = 1; side <= 2; ++side) {
    for (unsigned disk = 1; disk <= 3; ++disk) {
      std::ofstream* f = new std::ofstream(
          ("vertices_forward_" + std::to_string((side == 1 ? -(int(disk)) : (int)disk))).c_str(), std::ofstream::out);

      verticesFiles[1].push_back(f);
    }
  }

  for (auto& bin : bins) {
    unsigned rawId = bin.first;
    DetId id(rawId);
    unsigned subdetId = id.subdetId();

    if (subdetId != PixelSubdetector::PixelBarrel && subdetId != PixelSubdetector::PixelEndcap)
      continue;

    double* vertX = bin.second->GetX();
    double* vertY = bin.second->GetY();

    if (subdetId == PixelSubdetector::PixelBarrel) {
      PXBDetId pxbId(rawId);
      unsigned layer = tt->pxbLayer(pxbId);
      string onlineName = PixelBarrelName(pxbId, tt, true).name();

      *(verticesFiles[0][layer - 1]) << rawId << " " << onlineName << " \"";
      for (unsigned i = 0; i < 4; ++i) {
        *(verticesFiles[0][layer - 1]) << vertX[i] << "," << vertY[i];
        if (i == 3)
          *(verticesFiles[0][layer - 1]) << "\"\n";
        else
          *(verticesFiles[0][layer - 1]) << " ";
      }
    } else {
      PXFDetId pxfId(rawId);
      unsigned side = tt->pxfSide(pxfId);
      unsigned disk = tt->pxfDisk(pxfId);
      string onlineName = PixelEndcapName(pxfId, tt, true).name();
      unsigned mapIdx = disk + (side - 1) * 3 - 1;

      *(verticesFiles[1][mapIdx]) << rawId << " " << onlineName << " \"";
      for (unsigned i = 0; i < 3; ++i) {
        *(verticesFiles[1][mapIdx]) << vertX[i] << "," << vertY[i];

        if (i == 2)
          *(verticesFiles[1][mapIdx]) << "\"\n";
        else
          *(verticesFiles[1][mapIdx]) << " ";
      }
    }
  }

  for (unsigned i = 0; i < 2; ++i) {
    for (auto& j : verticesFiles[i]) {
      j->close();
      delete j;
    }
  }
}

void SiPixelPhase1Analyzer::FillBins(edm::Handle<reco::TrackCollection>* tracks,
                                     const TrackerGeometry& theTrackerGeometry,
                                     const TrackerTopology* tt) {
  switch (opMode) {
    case MODE_ANALYZE:
      for (auto const& track : *(*tracks)) {
        auto recHitsBegin = track.recHitsBegin();
        for (unsigned i = 0; i < track.recHitsSize(); ++i) {
          auto recHit = *(recHitsBegin + i);
          if (!recHit->isValid())
            continue;

          DetId id = recHit->geographicalId();
          unsigned subdetId = id.subdetId();

          if (subdetId != PixelSubdetector::PixelBarrel && subdetId != PixelSubdetector::PixelEndcap)
            continue;

          const PixelGeomDetUnit* geomdetunit =
              dynamic_cast<const PixelGeomDetUnit*>(theTrackerGeometry.idToDet(id));  // theTrackerGeometry ?????
          //const PixelTopology& topol = geomdetunit->specificTopology();

          LocalPoint localPoint = recHit->localPosition();
          GlobalPoint globalPoint = geomdetunit->surface().toGlobal(localPoint);

          if (subdetId == PixelSubdetector::PixelBarrel)
            FillBarrelBinsAnalyze(theTrackerGeometry, tt, id.rawId(), globalPoint);
          else
            FillForwardBinsAnalyze(theTrackerGeometry, tt, id.rawId(), globalPoint);
        }
      }
      break;
    case MODE_REMAP:
      FillBarrelBinsRemap(theTrackerGeometry, tt);
      FillForwardBinsRemap(theTrackerGeometry, tt);
      break;
    default:
      break;
  }
}

void SiPixelPhase1Analyzer::FillBarrelBinsAnalyze(const TrackerGeometry& theTrackerGeometry,
                                                  const TrackerTopology* tt,
                                                  unsigned rawId,
                                                  const GlobalPoint& globalPoint) {
  for (unsigned nameNum = 0; nameNum < baseHistogramName.size(); ++nameNum) {
    string strName = baseHistogramName[nameNum];

    PXBDetId id(rawId);

    int layer = tt->pxbLayer(id);

    th2PolyBarrel[strName][layer - 1]->Fill(TString::Format("%u", rawId), 1);
    th2PolyBarrelSummary[strName]->Fill(TString::Format("%u", rawId), 1);
#ifdef DEBUG_MODE
    th2PolyBarrelDebug[strName][layer - 1]->Fill((globalPoint.y() < 0 ? globalPoint.z() + 0.5f : globalPoint.z()),
                                                 -globalPoint.x(),
                                                 (globalPoint.y() < 0 ? -1 : 1));
#endif
  }
}

void SiPixelPhase1Analyzer::FillForwardBinsAnalyze(const TrackerGeometry& theTrackerGeometry,
                                                   const TrackerTopology* tt,
                                                   unsigned rawId,
                                                   const GlobalPoint& globalPoint) {
  for (unsigned nameNum = 0; nameNum < baseHistogramName.size(); ++nameNum) {
    string strName = baseHistogramName[nameNum];

    PXFDetId id(rawId);

    int side = tt->side(id);   //tt->pxfSide(id);
    int disk = tt->layer(id);  //tt->pxfDisk(id);
    // int blade = tt->pxfBlade(id);
    unsigned mapIdx = disk + (side - 1) * 3 - 1;

    pxfTh2PolyForward[strName][mapIdx]->Fill(TString::Format("%u", rawId), 1);
    pxfTh2PolyForwardSummary[strName]->Fill(TString::Format("%u", rawId), 1);

#ifdef DEBUG_MODE
    pxfTh2PolyForwardDebug[strName][mapIdx]->Fill(globalPoint.x(), globalPoint.y(), 1);
#endif
  }
}

void SiPixelPhase1Analyzer::FillBarrelBinsRemap(const TrackerGeometry& theTrackerGeometry, const TrackerTopology* tt) {
  rootFileHandle = new TFile(analazedRootFileName[0].c_str());

  if (!rootFileHandle) {
    LogInfo("Analyzer") << "Could not open file: " << analazedRootFileName[0] << "..." << endl;
    return;
  }

#ifdef DEBUG_MODE
  rootFileHandle->ls();
  LogInfo("Analyzer") << "\n\n";
  rootFileHandle->pwd();
  LogInfo("Analyzer") << "\n\n";
#endif

  for (unsigned nameNum = 0; nameNum < baseHistogramName.size(); ++nameNum) {
    if (!isBarrelSource[nameNum])
      continue;

    // if (pathToHistograms[nameNum][pathToHistograms.size() - 1] != '/') pathToHistograms[nameNum] += "/";
    string baseHistogramNameWithPath = pathToHistograms[nameNum] + baseHistogramName[nameNum] + "_";

    const TProfile2D* handles[4];
#ifndef DEBUG_MODE
    const TProfile2D* h;
#endif
    bool problemWithHandles = false;

    for (unsigned i = 0; i < 4; ++i) {
      string fullFileName = (baseHistogramNameWithPath + std::to_string(i + 1) + ";1 ");
      handles[i] = (TProfile2D*)rootFileHandle->Get(fullFileName.c_str());
      if (!handles[i]) {
        problemWithHandles = true;
        LogInfo("Analyzer") << "Histogram: " << fullFileName << " does not exist!\n";

        break;
      }
    }

    if (!problemWithHandles) {
      LogInfo("Analyzer") << "\nInput histograms: " << baseHistogramNameWithPath << " opened successfully\n";

      //Add original histograms to this file

      TDirectory* currDir = fs->getBareDirectory()->GetDirectory(baseHistogramName[nameNum].c_str());
      currDir->cd();

      for (unsigned i = 0; i < 4; ++i) {
        currDir->Add(handles[i]->Clone());
      }

      TrackingGeometry::DetContainer pxb = theTrackerGeometry.detsPXB();
#ifdef DEBUG_MODE
      debugFile << "There are " << pxb.size() << " detector elements in the PXB." << endl;
#endif
      for (auto& i : pxb) {
        const GeomDet* det = i;

        PXBDetId id = det->geographicalId();
        unsigned rawId = id.rawId();

        int module = tt->pxbModule(id);
        //int ladder = tt->pxbLadder(id);
        int layer = tt->pxbLayer(id);

        int signedOnlineModule = module - 4;
        if (signedOnlineModule <= 0)
          --signedOnlineModule;

        PixelBarrelName pixelBarrelName = PixelBarrelName(id, tt, true);
        int onlineShell = pixelBarrelName.shell();

        int signedOnlineLadder = ((onlineShell & 1) ? -pixelBarrelName.ladderName() : pixelBarrelName.ladderName());
        string strName = baseHistogramName[nameNum];

#ifdef DEBUG_MODE
        th2PolyBarrel[strName][layer - 1]->Fill(TString::Format("%u", rawId), signedOnlineLadder);
        th2PolyBarrelSummary[strName]->Fill(TString::Format("%u", rawId), signedOnlineLadder);
#else
        h = handles[layer - 1];
        unsigned nx = h->GetNbinsX();
        unsigned ny = h->GetNbinsY();
        unsigned binX = signedOnlineModule + ((nx + 1) >> 1);
        unsigned binY = (signedOnlineLadder) + ((ny + 1) >> 1);
        double val = h->GetBinContent(binX, binY);
        th2PolyBarrel[strName][layer - 1]->Fill(TString::Format("%u", rawId), val);
        th2PolyBarrelSummary[strName]->Fill(TString::Format("%u", rawId), val);
#endif
      }
    }
  }

  rootFileHandle->Close();
  delete rootFileHandle;
}

void SiPixelPhase1Analyzer::FillForwardBinsRemap(const TrackerGeometry& theTrackerGeometry, const TrackerTopology* tt) {
  rootFileHandle = new TFile(analazedRootFileName[0].c_str());

  if (!rootFileHandle) {
    return;
  }

  TrackingGeometry::DetContainer pxf = theTrackerGeometry.detsPXF();

#ifdef DEBUG_MODE
  rootFileHandle->ls();
  LogInfo("Analyzer") << "\n\n";
  rootFileHandle->pwd();
  LogInfo("Analyzer") << "\n\n";
#endif

  for (unsigned nameNum = 0; nameNum < baseHistogramName.size(); ++nameNum) {
    if (isBarrelSource[nameNum])
      continue;

    string baseHistogramNameWithPath = pathToHistograms[nameNum] + baseHistogramName[nameNum] + "_";

    const TProfile2D* h_1 = (TProfile2D*)rootFileHandle->Get((baseHistogramNameWithPath + "1;1 ").c_str());
    const TProfile2D* h_2 = (TProfile2D*)rootFileHandle->Get((baseHistogramNameWithPath + "2;1 ").c_str());
#ifndef DEBUG_MODE
    const TProfile2D* h;
#endif
    if (h_2 && h_1) {
      LogInfo("Analyzer") << "\nInput histograms: " << baseHistogramNameWithPath << " opened successfully\n";

      //Add original histograms to this file
      TDirectory* currDir = fs->getBareDirectory()->GetDirectory(baseHistogramName[nameNum].c_str());
      currDir->cd();
      currDir->Add(h_1->Clone());
      currDir->Add(h_2->Clone());

      for (auto& i : pxf) {
        const GeomDet* det = i;

        PXFDetId id = det->geographicalId();

        int side = tt->side(id);   //tt->pxfSide(id);
        int disk = tt->layer(id);  //tt->pxfDisk(id);

        unsigned rawId = id.rawId();
        PixelEndcapName pixelEndcapName = PixelEndcapName(PXFDetId(rawId), tt, true);

#ifdef DEBUG_MODE
        int blade = tt->pxfBlade(id);
#else
        int onlineBlade = pixelEndcapName.bladeName();
        bool isInnerOnlineBlade = !(pixelEndcapName.halfCylinder() & 1);  // inner -> blade > 0 (?)

        int signedOnlineBlade = (isInnerOnlineBlade) ? onlineBlade : -onlineBlade;
        int signedDisk = (side == 2) ? disk : -disk;

        int pannel = pixelEndcapName.pannelName() - 1;

#endif
        unsigned mapIdx = disk + (side - 1) * 3 - 1;
        string strName = baseHistogramName[nameNum];

#ifdef DEBUG_MODE
        pxfTh2PolyForward[strName][mapIdx]->Fill(TString::Format("%u", rawId), blade);
        pxfTh2PolyForwardSummary[strName]->Fill(TString::Format("%u", rawId), blade);
#else
        if (pixelEndcapName.ringName() == 1)
          h = h_1;
        else
          h = h_2;
        // ---- REMAP (Online -> Offline)
        unsigned nx = h->GetNbinsX();
        unsigned ny = h->GetNbinsY();
        unsigned binX = signedDisk + ((nx + 1) >> 1);
        unsigned binY = (signedOnlineBlade * 2) + (ny >> 1);
        double val = h->GetBinContent(binX, binY + pannel);
        pxfTh2PolyForward[strName][mapIdx]->Fill(TString::Format("%u", rawId), val);
        pxfTh2PolyForwardSummary[strName]->Fill(TString::Format("%u", rawId), val);
#endif
      }
    }
  }

  rootFileHandle->Close();
  delete rootFileHandle;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiPixelPhase1Analyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setComment(
      "Creates TH2Poly Pixel Tracker maps by either analyzing the event or remapping exising DQM historams");
  desc.add<edm::InputTag>("src", edm::InputTag("generalTracks"));
  desc.addUntracked<unsigned int>("opMode", 1);
  desc.addUntracked<std::string>("debugFileName", "debug.txt");
  desc.addUntracked<std::vector<unsigned int>>("isBarrelSource", {0, 0, 1});
  desc.addUntracked<std::vector<std::string>>("remapRootFileName", {"dqmFile.root"});
  desc.addUntracked<std::vector<std::string>>(
      "pathToHistograms",
      {"DQMData/Run 1/PixelPhase1/Run summary/Phase1_MechanicalView/PXForward/",
       "DQMData/Run 1/PixelPhase1/Run summary/Phase1_MechanicalView/PXForward/",
       "DQMData/Run 1/PixelPhase1/Run summary/Phase1_MechanicalView/PXBarrel/"});
  desc.addUntracked<std::vector<std::string>>("baseHistogramName",
                                              {"num_clusters_per_PXDisk_per_SignedBladePanel_PXRing",
                                               "num_digis_per_PXDisk_per_SignedBladePanel_PXRing",
                                               "num_digis_per_SignedModule_per_SignedLadder_PXLayer"});
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelPhase1Analyzer);
