#include "DQM/SiStripMonitorSummary/interface/SiStripCablingDQM.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "TCanvas.h"

SiStripCablingDQM::SiStripCablingDQM(edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> token,
                                     edm::RunNumber_t iRun,
                                     edm::ParameterSet const &hPSet,
                                     edm::ParameterSet const &fPSet,
                                     const TrackerTopology *tTopo,
                                     const TkDetMap *tkDetMap)
    : SiStripBaseCondObjDQMGet<SiStripDetCabling, SiStripDetCablingRcd>{token, iRun, hPSet, fPSet, tTopo} {
  if (HistoMaps_On_) {
    Tk_HM_ = std::make_unique<TkHistoMap>(tkDetMap, "SiStrip/Histo_Map", "Cabling_TkMap", 0.);
  }
}

SiStripCablingDQM::~SiStripCablingDQM() {}

void SiStripCablingDQM::getActiveDetIds(const edm::EventSetup &eSetup) {
  // Get active and total detIds
  getConditionObject(eSetup);
  if (!condObj_) {
    edm::LogError("InvalidCablingHandle") << "Invalid Cabling Handle";
    return;
  }
  condObj_->addActiveDetectorsRawIds(activeDetIds);
  condObj_->addAllDetectorsRawIds(activeDetIds);

  // Initialize arrays for counting:
  int counterTIB[4];
  for (int i = 0; i < 4; i++)
    counterTIB[i] = 0;
  int counterTID[2][3];
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++)
      counterTID[i][j] = 0;
  }
  int counterTOB[6];
  for (int i = 0; i < 6; i++)
    counterTOB[i] = 0;
  int counterTEC[2][9];
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 9; j++)
      counterTEC[i][j] = 0;
  }

  // fill arrays for counting and fill Histo_Map with value for connected :
  for (const auto detId : activeDetIds) {
    StripSubdetector subdet(detId);

    if (HistoMaps_On_) {
      Tk_HM_->fill(detId, condObj_->nApvPairs(detId) * 2);
    }
    if (fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")) {
      int32_t n_conn = 0;
      for (uint32_t connDet_i = 0; connDet_i < condObj_->getConnections(detId).size(); connDet_i++) {
        if (condObj_->getConnections(detId)[connDet_i] != nullptr &&
            condObj_->getConnections(detId)[connDet_i]->isConnected() != 0)
          n_conn++;
      }
      fillTkMap(detId, n_conn * 2.);
    }
    switch (subdet.subdetId()) {
      case StripSubdetector::TIB: {
        int i = tTopo_->tibLayer(detId) - 1;
        counterTIB[i]++;
        break;
      }
      case StripSubdetector::TID: {
        int j = tTopo_->tidWheel(detId) - 1;
        int side = tTopo_->tidSide(detId);
        if (side == 2) {
          counterTID[0][j]++;
        } else if (side == 1) {
          counterTID[1][j]++;
        }
        break;
      }
      case StripSubdetector::TOB: {
        int i = tTopo_->tobLayer(detId) - 1;
        counterTOB[i]++;
        break;
      }
      case StripSubdetector::TEC: {
        int j = tTopo_->tecWheel(detId) - 1;
        int side = tTopo_->tecSide(detId);
        if (side == 2) {
          counterTEC[0][j]++;
        } else if (side == 1) {
          counterTEC[1][j]++;
        }
        break;
      }
    }

  }  // idet

  // obtained from tracker.dat and hard-coded
  int TIBDetIds[4] = {672, 864, 540, 648};
  int TIDDetIds[2][3] = {{136, 136, 136}, {136, 136, 136}};
  int TOBDetIds[6] = {1008, 1152, 648, 720, 792, 888};
  int TECDetIds[2][9] = {{408, 408, 408, 360, 360, 360, 312, 312, 272}, {408, 408, 408, 360, 360, 360, 312, 312, 272}};

  DQMStore *dqmStore_ = edm::Service<DQMStore>().operator->();

  std::string FolderName = fPSet_.getParameter<std::string>("FolderName_For_QualityAndCabling_SummaryHistos");

  dqmStore_->setCurrentFolder(FolderName);

  //  dqmStore_->cd("SiStrip/MechanicalView/");
  MonitorElement *ME;
  ME = dqmStore_->book2D("SummaryOfCabling", "SummaryOfCabling", 6, 0.5, 6.5, 9, 0.5, 9.5);
  ME->setAxisTitle("Sub Det", 1);
  ME->setAxisTitle("Layer", 2);

  ME->setBinLabel(1, "TIB");
  ME->setBinLabel(2, "TID F");
  ME->setBinLabel(3, "TID B");
  ME->setBinLabel(4, "TOB");
  ME->setBinLabel(5, "TEC F");
  ME->setBinLabel(6, "TEC B");

  for (int i = 0; i < 4; i++) {
    ME->Fill(1, i + 1, float(counterTIB[i]) / TIBDetIds[i]);
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      ME->Fill(i + 2, j + 1, float(counterTID[i][j]) / TIDDetIds[i][j]);
    }
  }

  for (int i = 0; i < 6; i++) {
    ME->Fill(4, i + 1, float(counterTOB[i]) / TOBDetIds[i]);
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 9; j++) {
      ME->Fill(i + 5, j + 1, float(counterTEC[i][j]) / TECDetIds[i][j]);
    }
  }

  if (fPSet_.getParameter<bool>("OutputSummaryAtLayerLevelAsImage")) {
    TCanvas c1("c1");
    ME->getTH1()->Draw("TEXT");
    ME->getTH1()->SetStats(kFALSE);
    std::string name(ME->getTitle());
    name += ".png";
    c1.Print(name.c_str());
  }
}
