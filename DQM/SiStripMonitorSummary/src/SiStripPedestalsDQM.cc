#include "DQM/SiStripMonitorSummary/interface/SiStripPedestalsDQM.h"
#include "TCanvas.h"

SiStripPedestalsDQM::SiStripPedestalsDQM(edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> token,
                                         edm::RunNumber_t iRun,
                                         edm::ParameterSet const &hPSet,
                                         edm::ParameterSet const &fPSet,
                                         const TrackerTopology *tTopo,
                                         const TkDetMap *tkDetMap)
    : SiStripBaseCondObjDQMGet<SiStripPedestals, SiStripPedestalsRcd>{token, iRun, hPSet, fPSet, tTopo} {
  if (HistoMaps_On_) {
    Tk_HM_ = std::make_unique<TkHistoMap>(tkDetMap, "SiStrip/Histo_Map", "MeanPed_TkMap", 0.);
  }
}

SiStripPedestalsDQM::~SiStripPedestalsDQM() {}

void SiStripPedestalsDQM::getActiveDetIds(const edm::EventSetup &eSetup) {
  getConditionObject(eSetup);
  condObj_->getDetIds(activeDetIds);
}

void SiStripPedestalsDQM::fillModMEs(const std::vector<uint32_t> &selectedDetIds) {
  ModMEs CondObj_ME;
  for (const auto det : selectedDetIds) {
    fillMEsForDet(CondObj_ME, det);
  }
}

void SiStripPedestalsDQM::fillMEsForDet(const ModMEs &_selModME_, uint32_t selDetId_) {
  ModMEs selModME_ = _selModME_;
  getModMEs(selModME_, selDetId_);

  const auto pedRange = condObj_->getRange(selDetId_);
  int nStrip = detInfo_.getNumberOfApvsAndStripLength(selDetId_).first * 128;

  for (int istrip = 0; istrip < nStrip; ++istrip) {
    if (CondObj_fillId_ == "onlyProfile" || CondObj_fillId_ == "ProfileAndCumul") {
      selModME_.ProfileDistr->Fill(istrip + 1, condObj_->getPed(istrip, pedRange));
    }
  }  // istrip
}

void SiStripPedestalsDQM::fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) {
  for (const auto det : selectedDetIds) {
    fillMEsForLayer(/*SummaryMEsMap_,*/ det);
  }

  for (const auto &itm : SummaryMEsMap_) {
    ModMEs selME;
    selME = itm.second;

    if (hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel") &&
        fPSet_.getParameter<bool>("OutputSummaryProfileAtLayerLevelAsImage")) {
      if (CondObj_fillId_ == "onlyProfile" || CondObj_fillId_ == "ProfileAndCumul") {
        TCanvas c1("c1");
        selME.SummaryOfProfileDistr->getTProfile()->Draw();
        std::string name(selME.SummaryOfProfileDistr->getTProfile()->GetTitle());
        name += ".png";
        c1.Print(name.c_str());
      }
    }
    if (hPSet_.getParameter<bool>("FillSummaryAtLayerLevel") &&
        fPSet_.getParameter<bool>("OutputSummaryAtLayerLevelAsImage")) {
      TCanvas c1("c1");
      selME.SummaryDistr->getTH1()->Draw();
      std::string name(selME.SummaryDistr->getTitle());
      name += ".png";
      c1.Print(name.c_str());
    }
  }
}

void SiStripPedestalsDQM::fillMEsForLayer(
    /*std::map<uint32_t, ModMEs> selMEsMap_,*/ uint32_t selDetId_) {
  // ----
  int subdetectorId_ = ((selDetId_ >> 25) & 0x7);

  if (subdetectorId_ < 3 || subdetectorId_ > 6) {
    edm::LogError("SiStripPedestalsDQM") << "[SiStripPedestalsDQM::fillMEsForLayer] WRONG INPUT : no such "
                                            "subdetector type : "
                                         << subdetectorId_ << " no folder set!" << std::endl;
    return;
  }
  // ----

  //     // Cumulative distribution with average Ped value on a layer (not
  //     needed):

  const auto selMEsMapIter_ = SummaryMEsMap_.find(getLayerNameAndId(selDetId_).second);
  ModMEs selME_;
  if (selMEsMapIter_ != SummaryMEsMap_.end())
    selME_ = selMEsMapIter_->second;
  getSummaryMEs(selME_, selDetId_);

  const auto pedRange = condObj_->getRange(selDetId_);

  int nStrip = detInfo_.getNumberOfApvsAndStripLength(selDetId_).first * 128;

  SiStripHistoId hidmanager;

  if (hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel")) {
    // --> profile summary

    std::string hSummaryOfProfile_description;
    hSummaryOfProfile_description = hPSet_.getParameter<std::string>("SummaryOfProfile_description");

    std::string hSummaryOfProfile_name;

    hSummaryOfProfile_name =
        hidmanager.createHistoLayer(hSummaryOfProfile_description, "layer", getLayerNameAndId(selDetId_).first, "");

    for (int istrip = 0; istrip < nStrip; ++istrip) {
      if (CondObj_fillId_ == "onlyProfile" || CondObj_fillId_ == "ProfileAndCumul") {
        selME_.SummaryOfProfileDistr->Fill(istrip + 1, condObj_->getPed(istrip, pedRange));
      }

      // fill the TkMap
      if (fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")) {
        fillTkMap(selDetId_, condObj_->getPed(istrip, pedRange));
      }

    }  // istrip
  }    // if Fill ...

  if (hPSet_.getParameter<bool>("FillSummaryAtLayerLevel")) {
    // -->  summary

    std::string hSummary_description;
    hSummary_description = hPSet_.getParameter<std::string>("Summary_description");

    std::string hSummary_name;
    hSummary_name = hidmanager.createHistoLayer(hSummary_description, "layer", getLayerNameAndId(selDetId_).first, "");
    float meanPedestal = 0;

    for (int istrip = 0; istrip < nStrip; ++istrip) {
      meanPedestal = meanPedestal + condObj_->getPed(istrip, pedRange);

    }  // istrip

    meanPedestal = meanPedestal / nStrip;

    // -----
    // get detIds belonging to same layer to fill X-axis with detId-number

    std::vector<uint32_t> sameLayerDetIds_;

    sameLayerDetIds_.clear();

    sameLayerDetIds_ = GetSameLayerDetId(activeDetIds, selDetId_);

    unsigned int iBin = 0;
    for (unsigned int i = 0; i < sameLayerDetIds_.size(); i++) {
      if (sameLayerDetIds_[i] == selDetId_) {
        iBin = i + 1;
      }
    }

    selME_.SummaryDistr->Fill(iBin, meanPedestal);

    // Fill the Histo_TkMap with the mean Pedestal:
    if (HistoMaps_On_)
      Tk_HM_->fill(selDetId_, meanPedestal);

  }  // if Fill ...
}
