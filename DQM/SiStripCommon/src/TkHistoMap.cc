#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"

//#define debug_TkHistoMap

TkHistoMap::TkHistoMap(const TkDetMap* tkDetMap) : HistoNumber(35) {
  LogTrace("TkHistoMap") << "TkHistoMap::constructor without parameters";
  load(tkDetMap, "", 0.0f, false, false, false);
}

TkHistoMap::TkHistoMap(
    const TkDetMap* tkDetMap, const std::string& path, const std::string& MapName, float baseline, bool mechanicalView)
    : HistoNumber(35), MapName_(MapName) {
  LogTrace("TkHistoMap") << "TkHistoMap::constructor with parameters";
  load(tkDetMap, path, baseline, mechanicalView, false);
  dqmStore_->meBookerGetter([this, &path, &baseline, mechanicalView](DQMStore::IBooker& ibooker, DQMStore::IGetter&) {
    this->createTkHistoMap(ibooker, path, MapName_, baseline, mechanicalView);
  });
}

TkHistoMap::TkHistoMap(const TkDetMap* tkDetMap,
                       const std::string& path,
                       const std::string& MapName,
                       float baseline,
                       bool mechanicalView,
                       bool isTH2F)
    : HistoNumber(35), MapName_(MapName) {
  LogTrace("TkHistoMap") << "TkHistoMap::constructor with parameters";
  load(tkDetMap, path, baseline, mechanicalView, isTH2F);
  dqmStore_->meBookerGetter([this, &path, &baseline, mechanicalView](DQMStore::IBooker& ibooker, DQMStore::IGetter&) {
    this->createTkHistoMap(ibooker, path, MapName_, baseline, mechanicalView);
  });
}

TkHistoMap::TkHistoMap(const TkDetMap* tkDetMap,
                       DQMStore::IBooker& ibooker,
                       const std::string& path,
                       const std::string& MapName,
                       float baseline,
                       bool mechanicalView,
                       bool isTH2F)
    : HistoNumber(35), MapName_(MapName) {
  LogTrace("TkHistoMap") << "TkHistoMap::constructor with parameters";
  load(tkDetMap, path, baseline, mechanicalView, isTH2F);
  createTkHistoMap(ibooker, path, MapName_, baseline, mechanicalView);
}

void TkHistoMap::load(const TkDetMap* tkDetMap,
                      const std::string& path,
                      float baseline,
                      bool mechanicalView,
                      bool isTH2F,
                      bool createTkMap) {
  // cannot pass nullptr, otherwise methods making use of TrackerTopology will segfault
  if (tkDetMap == nullptr) {
    throw cms::Exception("LogicError") << " expected pointer to TkDetMap is null!\n";
  }

  cached_detid = 0;
  cached_layer = 0;
  loadServices();
  tkdetmap_ = tkDetMap;
  isTH2F_ = isTH2F;
}

void TkHistoMap::loadServices() {
  if (!edm::Service<DQMStore>().isAvailable()) {
    edm::LogError("TkHistoMap")
        << "\n------------------------------------------"
           "\nUnAvailable Service DQMStore: please insert in the configuration file an instance like"
           "\n\tprocess.load(\"DQMServices.Core.DQMStore_cfg\")"
           "\n------------------------------------------";
  }
  dqmStore_ = edm::Service<DQMStore>().operator->();
}

void TkHistoMap::save(const std::string& filename) {
  // dqmStore_ only for saving
  dqmStore_->save(filename);
}

void TkHistoMap::loadTkHistoMap(const std::string& path, const std::string& MapName, bool mechanicalView) {
  MapName_ = MapName;
  tkHistoMap_.resize(HistoNumber);
  auto loadMap = [this, &path, mechanicalView](DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
    std::string fullName, folder;
    for (int layer = 1; layer < HistoNumber; ++layer) {
      folder = folderDefinition(ibooker, path, MapName_, layer, mechanicalView, fullName);
#ifdef debug_TkHistoMap
      LogTrace("TkHistoMap") << "[TkHistoMap::loadTkHistoMap] folder " << folder << " histoName " << fullName
                             << " find " << folder.find_last_of("/") << "  length " << folder.length();
#endif
      if (folder.find_last_of('/') != folder.length() - 1)
        folder += "/";
      tkHistoMap_[layer] = igetter.get(folder + fullName);
#ifdef debug_TkHistoMap
      LogTrace("TkHistoMap") << "[TkHistoMap::loadTkHistoMap] folder " << folder << " histoName " << fullName
                             << " layer " << layer << " ptr " << tkHistoMap_[layer] << " find "
                             << folder.find_last_of("/") << "  length " << folder.length();
#endif
    }
  };
  dqmStore_->meBookerGetter(loadMap);
}

void TkHistoMap::createTkHistoMap(DQMStore::IBooker& ibooker,
                                  const std::string& path,
                                  const std::string& MapName,
                                  float baseline,
                                  bool mechanicalView) {
  int nchX;
  int nchY;
  double lowX, highX;
  double lowY, highY;
  std::string fullName, folder;

  tkHistoMap_.resize(HistoNumber);
  const bool bookTH2F = isTH2F_;
  for (int layer = 1; layer < HistoNumber; ++layer) {
    folder = folderDefinition(ibooker, path, MapName, layer, mechanicalView, fullName);
    tkdetmap_->getComponents(layer, nchX, lowX, highX, nchY, lowY, highY);
    MonitorElement* me;
    if (bookTH2F == false) {
      me = ibooker.bookProfile2D(fullName.c_str(), fullName.c_str(), nchX, lowX, highX, nchY, lowY, highY, 0.0, 0.0);
    } else {
      me = ibooker.book2D(fullName.c_str(), fullName.c_str(), nchX, lowX, highX, nchY, lowY, highY);
    }
    //initialize bin content for the not assigned bins
    if (baseline != 0) {
      for (size_t ix = 1; ix <= (unsigned int)nchX; ++ix)
        for (size_t iy = 1; iy <= (unsigned int)nchY; ++iy)
          if (!tkdetmap_->getDetFromBin(layer, ix, iy))
            me->Fill(1. * (lowX + ix - .5), 1. * (lowY + iy - .5), baseline);
    }

    tkHistoMap_[layer] = me;
#ifdef debug_TkHistoMap
    LogTrace("TkHistoMap") << "[TkHistoMap::createTkHistoMap] folder " << folder << " histoName " << fullName
                           << " layer " << layer << " ptr " << tkHistoMap_[layer];
#endif
  }
}

std::string TkHistoMap::folderDefinition(DQMStore::IBooker& ibooker,
                                         std::string folder,
                                         const std::string& MapName,
                                         int layer,
                                         bool mechanicalView,
                                         std::string& fullName) {
  std::string name = MapName + std::string("_");
  fullName = name + TkDetMap::getLayerName(layer);

  if (mechanicalView) {
    std::stringstream ss;

    SiStripFolderOrganizer folderOrg;
    folderOrg.setSiStripFolderName(folder);

    SiStripDetId::SubDetector subDet;
    uint32_t subdetlayer = 0, side = 0;
    TkDetMap::getSubDetLayerSide(layer, subDet, subdetlayer, side);
    folderOrg.getSubDetLayerFolderName(ss, subDet, subdetlayer, side);
    folder = ss.str();
  }
  ibooker.setCurrentFolder(folder);
  return folder;
}

#include <iostream>
void TkHistoMap::fillFromAscii(const std::string& filename) {
  std::ifstream file;
  file.open(filename.c_str());

  if (file.fail()) {
    throw cms::Exception("LogicError") << "failed to open input file" << std::endl;
  }

  float value;
  uint32_t detid;
  while (file.good()) {
    file >> detid >> value;
    fill(detid, value);
  }
  file.close();
}

void TkHistoMap::fill(DetId detid, float value) {
  int16_t layer = tkdetmap_->findLayer(detid, cached_detid, cached_layer, cached_XYbin);
  if (layer == TkLayerMap::INVALID) {
    edm::LogError("TkHistoMap") << " could not fill for detid " << detid.rawId() << ", as the layer is invalid";
    return;
  }

  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid, cached_detid, cached_layer, cached_XYbin);
#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::fill] Fill detid " << detid.rawId() << " Layer " << layer << " value "
                         << value << " ix,iy " << xybin.ix << " " << xybin.iy << " " << xybin.x << " " << xybin.y << " "
                         << tkHistoMap_[layer]->getTProfile2D()->GetName();
#endif
  if (tkHistoMap_[layer]->kind() == MonitorElement::Kind::TPROFILE2D)
    tkHistoMap_[layer]->getTProfile2D()->Fill(xybin.x, xybin.y, value);
  else if (tkHistoMap_[layer]->kind() == MonitorElement::Kind::TH2F)
    tkHistoMap_[layer]->getTH2F()->Fill(xybin.x, xybin.y, value);
#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::fill] "
                         << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(xybin.ix, xybin.iy);
  for (size_t ii = 0; ii < 4; ii++)
    for (size_t jj = 0; jj < 11; jj++) {
      if (tkHistoMap_[layer]->kind() == MonitorElement::Kind::TPROFILE2D)
        LogTrace("TkHistoMap") << "[TkHistoMap::fill] " << ii << " " << jj << " "
                               << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(ii, jj);
      if (tkHistoMap_[layer]->kind() == MonitorElement::Kind::TH2F)
        LogTrace("TkHistoMap") << "[TkHistoMap::fill] " << ii << " " << jj << " "
                               << tkHistoMap_[layer]->getTH2F()->GetBinContent(ii, jj);
    }
#endif
}

void TkHistoMap::setBinContent(DetId detid, float value) {
  int16_t layer = tkdetmap_->findLayer(detid, cached_detid, cached_layer, cached_XYbin);
  if (layer == TkLayerMap::INVALID) {
    edm::LogError("TkHistoMap") << " could not setBinContent for detid " << detid.rawId()
                                << ", as the layer is invalid";
    return;
  }
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid, cached_detid, cached_layer, cached_XYbin);

  if (tkHistoMap_[layer]->kind() == MonitorElement::Kind::TPROFILE2D) {
    tkHistoMap_[layer]->getTProfile2D()->SetBinEntries(tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix, xybin.iy),
                                                       1);
    tkHistoMap_[layer]->getTProfile2D()->SetBinContent(tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix, xybin.iy),
                                                       value);
  } else if (tkHistoMap_[layer]->kind() == MonitorElement::Kind::TH2F) {
    tkHistoMap_[layer]->setBinContent(xybin.ix, xybin.iy, value);
  }

#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent]  setBinContent detid " << detid.rawId() << " Layer " << layer
                         << " value " << value << " ix,iy " << xybin.ix << " " << xybin.iy << " " << xybin.x << " "
                         << xybin.y << " " << tkHistoMap_[layer]->getTProfile2D()->GetName() << " bin "
                         << tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix, xybin.iy);

  LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent] "
                         << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(xybin.ix, xybin.iy);
  for (size_t ii = 0; ii < 4; ii++)
    for (size_t jj = 0; jj < 11; jj++) {
      LogTrace("TkHistoMap") << "[TkHistoMap::setbincontent] " << ii << " " << jj << " "
                             << tkHistoMap_[layer]->getTProfile2D()->GetBinContent(ii, jj);
    }
#endif
}

void TkHistoMap::add(DetId detid, float value) {
#ifdef debug_TkHistoMap
  LogTrace("TkHistoMap") << "[TkHistoMap::add]";
#endif
  int16_t layer = tkdetmap_->findLayer(detid, cached_detid, cached_layer, cached_XYbin);
  if (layer == TkLayerMap::INVALID) {
    edm::LogError("TkHistoMap") << " could not add for detid " << detid.rawId() << ", as the layer is invalid";
    return;
  }
  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid, cached_detid, cached_layer, cached_XYbin);
  if (tkHistoMap_[layer]->kind() == MonitorElement::Kind::TPROFILE2D)
    setBinContent(detid,
                  tkHistoMap_[layer]->getTProfile2D()->GetBinContent(
                      tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix, xybin.iy)) +
                      value);
  else if (tkHistoMap_[layer]->kind() == MonitorElement::Kind::TH2F)
    setBinContent(
        detid,
        tkHistoMap_[layer]->getTH2F()->GetBinContent(tkHistoMap_[layer]->getTH2F()->GetBin(xybin.ix, xybin.iy)) +
            value);
}

float TkHistoMap::getValue(DetId detid) {
  int16_t layer = tkdetmap_->findLayer(detid, cached_detid, cached_layer, cached_XYbin);
  if (layer == TkLayerMap::INVALID) {
    edm::LogError("TkHistoMap") << " could not getValue for detid " << detid.rawId() << ", as the layer is invalid";
    return -99999.f;
  }

  TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid, cached_detid, cached_layer, cached_XYbin);

  if (tkHistoMap_[layer]->kind() == MonitorElement::Kind::TH2F)
    return tkHistoMap_[layer]->getTH2F()->GetBinContent(tkHistoMap_[layer]->getTH2F()->GetBin(xybin.ix, xybin.iy));
  else
    return tkHistoMap_[layer]->getTProfile2D()->GetBinContent(
        tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix, xybin.iy));
}
float TkHistoMap::getEntries(DetId detid) {
  int16_t layer = tkdetmap_->findLayer(detid, cached_detid, cached_layer, cached_XYbin);
  if (layer == TkLayerMap::INVALID) {
    edm::LogError("TkHistoMap") << " could not getValue for detid " << detid.rawId() << ", as the layer is invalid";
    return -99999.f;
  }

  if (tkHistoMap_[layer]->kind() == MonitorElement::Kind::TH2F) {
    return 1;
  } else {
    TkLayerMap::XYbin xybin = tkdetmap_->getXY(detid, cached_detid, cached_layer, cached_XYbin);
    return tkHistoMap_[layer]->getTProfile2D()->GetBinEntries(
        tkHistoMap_[layer]->getTProfile2D()->GetBin(xybin.ix, xybin.iy));
  }
}

void TkHistoMap::dumpInTkMap(TrackerMap* tkmap, bool dumpEntries) {
  for (int layer = 1; layer < HistoNumber; ++layer) {
    // std::vector<uint32_t> dets;
    // tkdetmap_->getDetsForLayer(layer,dets);
    std::vector<DetId> dets = tkdetmap_->getDetsForLayer(layer);
    for (size_t i = 0; i < dets.size(); ++i) {
      if (dets[i] > 0) {
        if (getEntries(dets[i]) > 0) {
          tkmap->fill(dets[i], dumpEntries ? getEntries(dets[i]) : getValue(dets[i]));
        }
      }
    }
  }
}

#include "TCanvas.h"
#include "TFile.h"
void TkHistoMap::saveAsCanvas(const std::string& filename, const std::string& options, const std::string& mode) {
  //  TCanvas C(MapName_,MapName_,200,10,900,700);
  TCanvas* CTIB =
      new TCanvas(std::string("Canvas_" + MapName_ + "TIB").c_str(), std::string("Canvas_" + MapName_ + "TIB").c_str());
  TCanvas* CTOB =
      new TCanvas(std::string("Canvas_" + MapName_ + "TOB").c_str(), std::string("Canvas_" + MapName_ + "TOB").c_str());
  TCanvas* CTIDP = new TCanvas(std::string("Canvas_" + MapName_ + "TIDP").c_str(),
                               std::string("Canvas_" + MapName_ + "TIDP").c_str());
  TCanvas* CTIDM = new TCanvas(std::string("Canvas_" + MapName_ + "TIDM").c_str(),
                               std::string("Canvas_" + MapName_ + "TIDM").c_str());
  TCanvas* CTECP = new TCanvas(std::string("Canvas_" + MapName_ + "TECP").c_str(),
                               std::string("Canvas_" + MapName_ + "TECP").c_str());
  TCanvas* CTECM = new TCanvas(std::string("Canvas_" + MapName_ + "TECM").c_str(),
                               std::string("Canvas_" + MapName_ + "TECM").c_str());
  CTIB->Divide(2, 2);
  CTOB->Divide(2, 3);
  CTIDP->Divide(1, 3);
  CTIDM->Divide(1, 3);
  CTECP->Divide(3, 3);
  CTECM->Divide(3, 3);

  int i;
  i = 0;
  CTIB->cd(++i);
  tkHistoMap_[TkLayerMap::TIB_L1]->getTProfile2D()->Draw(options.c_str());
  CTIB->cd(++i);
  tkHistoMap_[TkLayerMap::TIB_L2]->getTProfile2D()->Draw(options.c_str());
  CTIB->cd(++i);
  tkHistoMap_[TkLayerMap::TIB_L3]->getTProfile2D()->Draw(options.c_str());
  CTIB->cd(++i);
  tkHistoMap_[TkLayerMap::TIB_L4]->getTProfile2D()->Draw(options.c_str());

  i = 0;
  CTIDP->cd(++i);
  tkHistoMap_[TkLayerMap::TIDP_D1]->getTProfile2D()->Draw(options.c_str());
  CTIDP->cd(++i);
  tkHistoMap_[TkLayerMap::TIDP_D2]->getTProfile2D()->Draw(options.c_str());
  CTIDP->cd(++i);
  tkHistoMap_[TkLayerMap::TIDP_D3]->getTProfile2D()->Draw(options.c_str());

  i = 0;
  CTIDM->cd(++i);
  tkHistoMap_[TkLayerMap::TIDM_D1]->getTProfile2D()->Draw(options.c_str());
  CTIDM->cd(++i);
  tkHistoMap_[TkLayerMap::TIDM_D2]->getTProfile2D()->Draw(options.c_str());
  CTIDM->cd(++i);
  tkHistoMap_[TkLayerMap::TIDM_D3]->getTProfile2D()->Draw(options.c_str());

  i = 0;
  CTOB->cd(++i);
  tkHistoMap_[TkLayerMap::TOB_L1]->getTProfile2D()->Draw(options.c_str());
  CTOB->cd(++i);
  tkHistoMap_[TkLayerMap::TOB_L2]->getTProfile2D()->Draw(options.c_str());
  CTOB->cd(++i);
  tkHistoMap_[TkLayerMap::TOB_L3]->getTProfile2D()->Draw(options.c_str());
  CTOB->cd(++i);
  tkHistoMap_[TkLayerMap::TOB_L4]->getTProfile2D()->Draw(options.c_str());
  CTOB->cd(++i);
  tkHistoMap_[TkLayerMap::TOB_L5]->getTProfile2D()->Draw(options.c_str());
  CTOB->cd(++i);
  tkHistoMap_[TkLayerMap::TOB_L6]->getTProfile2D()->Draw(options.c_str());

  i = 0;
  CTECP->cd(++i);
  tkHistoMap_[TkLayerMap::TECP_W1]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);
  tkHistoMap_[TkLayerMap::TECP_W2]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);
  tkHistoMap_[TkLayerMap::TECP_W3]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);
  tkHistoMap_[TkLayerMap::TECP_W4]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);
  tkHistoMap_[TkLayerMap::TECP_W5]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);
  tkHistoMap_[TkLayerMap::TECP_W6]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);
  tkHistoMap_[TkLayerMap::TECP_W7]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);
  tkHistoMap_[TkLayerMap::TECP_W8]->getTProfile2D()->Draw(options.c_str());
  CTECP->cd(++i);
  tkHistoMap_[TkLayerMap::TECP_W9]->getTProfile2D()->Draw(options.c_str());

  i = 0;
  CTECM->cd(++i);
  tkHistoMap_[TkLayerMap::TECM_W1]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);
  tkHistoMap_[TkLayerMap::TECM_W2]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);
  tkHistoMap_[TkLayerMap::TECM_W3]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);
  tkHistoMap_[TkLayerMap::TECM_W4]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);
  tkHistoMap_[TkLayerMap::TECM_W5]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);
  tkHistoMap_[TkLayerMap::TECM_W6]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);
  tkHistoMap_[TkLayerMap::TECM_W7]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);
  tkHistoMap_[TkLayerMap::TECM_W8]->getTProfile2D()->Draw(options.c_str());
  CTECM->cd(++i);
  tkHistoMap_[TkLayerMap::TECM_W9]->getTProfile2D()->Draw(options.c_str());

  TFile* f = new TFile(filename.c_str(), mode.c_str());
  CTIB->Write();
  CTIDP->Write();
  CTIDM->Write();
  CTOB->Write();
  CTECP->Write();
  CTECM->Write();
  f->Close();
  delete f;
}
