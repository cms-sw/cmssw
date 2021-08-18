#include "DQM/SiStripMonitorSummary/interface/SiStripThresholdDQM.h"

SiStripThresholdDQM::SiStripThresholdDQM(edm::ESGetToken<SiStripThreshold, SiStripThresholdRcd> token,
                                         edm::RunNumber_t iRun,
                                         edm::ParameterSet const &hPSet,
                                         edm::ParameterSet const &fPSet,
                                         const TrackerTopology *tTopo,
                                         const TkDetMap *tkDetMap)
    : SiStripBaseCondObjDQMGet<SiStripThreshold, SiStripThresholdRcd>{token, iRun, hPSet, fPSet, tTopo} {
  WhichThreshold = hPSet.getParameter<std::string>("WhichThreshold");

  if (HistoMaps_On_) {
    if (WhichThreshold == "Low")
      Tk_HM_L = std::make_unique<TkHistoMap>(tkDetMap, "SiStrip/Histo_Map", "LowThresh_TkMap", 0.);
    if (WhichThreshold == "High")
      Tk_HM_H = std::make_unique<TkHistoMap>(tkDetMap, "SiStrip/Histo_Map", "HighThresh_TkMap", 0.);
  }
}

SiStripThresholdDQM::~SiStripThresholdDQM() {}

void SiStripThresholdDQM::getActiveDetIds(const edm::EventSetup &eSetup) {
  getConditionObject(eSetup);
  condObj_->getDetIds(activeDetIds);
}

void SiStripThresholdDQM::fillModMEs(const std::vector<uint32_t> &selectedDetIds) {
  ModMEs CondObj_ME;
  for (const auto det : selectedDetIds) {
    fillMEsForDet(CondObj_ME, det);
  }
}

void SiStripThresholdDQM::fillMEsForDet(const ModMEs &_selModME_, uint32_t selDetId_) {
  ModMEs selModME_ = _selModME_;
  std::vector<uint32_t> DetIds;
  condObj_->getDetIds(DetIds);

  const auto ThresholdRange = condObj_->getRange(selDetId_);
  int nStrip = detInfo_.getNumberOfApvsAndStripLength(selDetId_).first * 128;

  getModMEs(selModME_, selDetId_);

  for (int istrip = 0; istrip < nStrip; ++istrip) {
    if (CondObj_fillId_ == "onlyProfile" || CondObj_fillId_ == "ProfileAndCumul") {
      if (WhichThreshold == "Low")
        selModME_.ProfileDistr->Fill(istrip + 1, condObj_->getData(istrip, ThresholdRange).getLth());
      if (WhichThreshold == "High")
        selModME_.ProfileDistr->Fill(istrip + 1, condObj_->getData(istrip, ThresholdRange).getHth());
    }
  }  // istrip
}

void SiStripThresholdDQM::fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) {
  for (const auto det : selectedDetIds) {
    fillMEsForLayer(/*SummaryMEsMap_,*/ det);
  }
}

void SiStripThresholdDQM::fillMEsForLayer(
    /*std::map<uint32_t, ModMEs> selMEsMap_,*/ uint32_t selDetId_) {
  // ----
  int subdetectorId_ = ((selDetId_ >> 25) & 0x7);

  if (subdetectorId_ < 3 || subdetectorId_ > 6) {
    edm::LogError("SiStripThresholdDQM") << "[SiStripThresholdDQM::fillMEsForLayer] WRONG INPUT : no such "
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

  const auto ThresholdRange = condObj_->getRange(selDetId_);

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
        if (WhichThreshold == "Low") {
          selME_.SummaryOfProfileDistr->Fill(istrip + 1, condObj_->getData(istrip, ThresholdRange).getLth());
          if (fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On"))
            fillTkMap(selDetId_, condObj_->getData(istrip, ThresholdRange).getLth());
        }
        if (WhichThreshold == "High") {
          selME_.SummaryOfProfileDistr->Fill(istrip + 1, condObj_->getData(istrip, ThresholdRange).getHth());
          if (fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On"))
            fillTkMap(selDetId_, condObj_->getData(istrip, ThresholdRange).getHth());
        }
      }
    }  // istrip

  }  // if Fill

  if (hPSet_.getParameter<bool>("FillSummaryAtLayerLevel")) {
    // --> summary

    std::string hSummary_description;
    hSummary_description = hPSet_.getParameter<std::string>("Summary_description");

    std::string hSummary_name;
    hSummary_name = hidmanager.createHistoLayer(hSummary_description, "layer", getLayerNameAndId(selDetId_).first, "");

    float meanLowThreshold = 0;
    float meanHighThreshold = 0;

    for (int istrip = 0; istrip < nStrip; ++istrip) {
      meanLowThreshold = meanLowThreshold + condObj_->getData(istrip, ThresholdRange).getLth();
      meanHighThreshold = meanHighThreshold + condObj_->getData(istrip, ThresholdRange).getHth();
    }  // istrip

    meanLowThreshold = meanLowThreshold / nStrip;
    meanHighThreshold = meanHighThreshold / nStrip;

    // -----
    // get detIds belonging to same layer to fill X-axis with detId-number

    std::vector<uint32_t> sameLayerDetIds_ = GetSameLayerDetId(activeDetIds, selDetId_);

    unsigned int iBin = 0;
    for (unsigned int i = 0; i < sameLayerDetIds_.size(); i++) {
      if (sameLayerDetIds_[i] == selDetId_) {
        iBin = i + 1;
      }
    }

    if (WhichThreshold == "Low") {
      selME_.SummaryDistr->Fill(iBin, meanLowThreshold);
      if (HistoMaps_On_)
        Tk_HM_L->fill(selDetId_, meanLowThreshold);
    }
    if (WhichThreshold == "High") {
      selME_.SummaryDistr->Fill(iBin, meanHighThreshold);
      if (HistoMaps_On_)
        Tk_HM_H->fill(selDetId_, meanHighThreshold);
    }

  }  // if Fill ...
}
