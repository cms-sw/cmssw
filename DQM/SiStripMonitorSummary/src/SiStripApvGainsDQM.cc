#include "DQM/SiStripMonitorSummary/interface/SiStripApvGainsDQM.h"

#include "TCanvas.h"

SiStripApvGainsDQM::SiStripApvGainsDQM(edm::ESGetToken<SiStripApvGain, SiStripApvGainRcd> token,
                                       edm::RunNumber_t iRun,
                                       edm::ParameterSet const &hPSet,
                                       edm::ParameterSet const &fPSet,
                                       const TrackerTopology *tTopo,
                                       const TkDetMap *tkDetMap)
    : SiStripBaseCondObjDQMGet<SiStripApvGain, SiStripApvGainRcd>{token, iRun, hPSet, fPSet, tTopo} {
  if (HistoMaps_On_) {
    Tk_HM_ = std::make_unique<TkHistoMap>(tkDetMap, "SiStrip/Histo_Map", "MeanApvGain_TkMap", 0.);
  }
}

SiStripApvGainsDQM::~SiStripApvGainsDQM() {}

void SiStripApvGainsDQM::getActiveDetIds(const edm::EventSetup &eSetup) {
  getConditionObject(eSetup);
  condObj_->getDetIds(activeDetIds);
}

void SiStripApvGainsDQM::fillModMEs(const std::vector<uint32_t> &selectedDetIds) {
  ModMEs CondObj_ME;
  for (const auto detIter : selectedDetIds) {
    fillMEsForDet(CondObj_ME, detIter);
  }
}

void SiStripApvGainsDQM::fillMEsForDet(const ModMEs &_selModME_, uint32_t selDetId_) {
  ModMEs selModME_ = _selModME_;
  std::vector<uint32_t> DetIds;
  condObj_->getDetIds(DetIds);

  const auto gainRange = condObj_->getRange(selDetId_);

  int nApv = detInfo_.getNumberOfApvsAndStripLength(selDetId_).first;

  getModMEs(selModME_, selDetId_);

  for (int iapv = 0; iapv < nApv; ++iapv) {
    if (CondObj_fillId_ == "onlyProfile" || CondObj_fillId_ == "ProfileAndCumul") {
      selModME_.ProfileDistr->Fill(iapv + 1, condObj_->getApvGain(iapv, gainRange));
    }
    if (CondObj_fillId_ == "onlyCumul" || CondObj_fillId_ == "ProfileAndCumul") {
      selModME_.CumulDistr->Fill(condObj_->getApvGain(iapv, gainRange));
    }

    // Fill the TkMap
    if (fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")) {
      fillTkMap(selDetId_, condObj_->getApvGain(iapv, gainRange));
    }
  }
}

void SiStripApvGainsDQM::fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) {
  for (const auto det : selectedDetIds) {
    fillMEsForLayer(/*SummaryMEsMap_,*/ det);
  }

  for (const auto &itm : SummaryMEsMap_) {
    ModMEs selME = itm.second;
    if (hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel") &&
        fPSet_.getParameter<bool>("OutputSummaryProfileAtLayerLevelAsImage")) {
      TCanvas c1("c1");
      selME.SummaryOfProfileDistr->getTProfile()->Draw();
      std::string name(selME.SummaryOfProfileDistr->getTProfile()->GetTitle());
      name += ".png";
      c1.Print(name.c_str());
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

void SiStripApvGainsDQM::fillMEsForLayer(
    /*std::map<uint32_t, ModMEs> selMEsMap_, */ uint32_t selDetId_) {
  int subdetectorId_ = ((selDetId_ >> 25) & 0x7);

  if (subdetectorId_ < 3 || subdetectorId_ > 6) {
    edm::LogError("SiStripApvGainsDQM") << "[SiStripApvGainsDQM::fillMEsForLayer] WRONG INPUT : no such "
                                           "subdetector type : "
                                        << subdetectorId_ << " no folder set!" << std::endl;
    return;
  }
  // ----

  const auto selMEsMapIter_ = SummaryMEsMap_.find(getLayerNameAndId(selDetId_).second);
  ModMEs selME_;
  if (selMEsMapIter_ != SummaryMEsMap_.end())
    selME_ = selMEsMapIter_->second;
  getSummaryMEs(selME_, selDetId_);

  const auto gainRange = condObj_->getRange(selDetId_);
  int nApv = detInfo_.getNumberOfApvsAndStripLength(selDetId_).first;

  float meanApvGain = 0;

  SiStripHistoId hidmanager;

  if (hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel")) {
    // --> profile summary
    std::string hSummaryOfProfile_description;
    hSummaryOfProfile_description = hPSet_.getParameter<std::string>("SummaryOfProfile_description");

    std::string hSummaryOfProfile_name;
    hSummaryOfProfile_name =
        hidmanager.createHistoLayer(hSummaryOfProfile_description, "layer", getLayerNameAndId(selDetId_).first, "");

    for (int iapv = 0; iapv < nApv; ++iapv) {
      meanApvGain = meanApvGain + condObj_->getApvGain(iapv, gainRange);
      selME_.SummaryOfProfileDistr->Fill(iapv + 1, condObj_->getApvGain(iapv, gainRange));

      // Fill the TkMap
      if (fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")) {
        fillTkMap(selDetId_, condObj_->getApvGain(iapv, gainRange));
      }

    }  // iapv

    meanApvGain = meanApvGain / nApv;

    // Fill the TkHistoMap with meanApvgain:
    if (HistoMaps_On_)
      Tk_HM_->setBinContent(selDetId_, meanApvGain);

  }  // if Fill ...

  if (hPSet_.getParameter<bool>("FillSummaryAtLayerLevel")) {
    // -->  summary
    std::string hSummary_description;
    hSummary_description = hPSet_.getParameter<std::string>("Summary_description");

    std::string hSummary_name;
    hSummary_name = hidmanager.createHistoLayer(hSummary_description, "layer", getLayerNameAndId(selDetId_).first, "");

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

    for (int iapv = 0; iapv < nApv; ++iapv) {
      meanApvGain = meanApvGain + condObj_->getApvGain(iapv, gainRange);
      selME_.SummaryDistr->Fill(iBin, condObj_->getApvGain(iapv, gainRange));
    }  // iapv
    meanApvGain = meanApvGain / nApv;

    // Fill the TkHistoMap with meanApvgain:
    //    if(HistoMaps_On_ ) Tk_HM_->setBinContent(selDetId_, meanApvGain);

  }  // if Fill ...
}
