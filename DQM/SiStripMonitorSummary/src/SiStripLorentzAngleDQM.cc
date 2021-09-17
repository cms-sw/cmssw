#include "DQM/SiStripMonitorSummary/interface/SiStripLorentzAngleDQM.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/SiStripSubStructure.h"
#include "TCanvas.h"

SiStripLorentzAngleDQM::SiStripLorentzAngleDQM(edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> token,
                                               edm::RunNumber_t iRun,
                                               edm::ParameterSet const &hPSet,
                                               edm::ParameterSet const &fPSet,
                                               const TrackerTopology *tTopo,
                                               const TkDetMap *tkDetMap)
    : SiStripBaseCondObjDQMGet<SiStripLorentzAngle, SiStripLorentzAngleRcd>{token, iRun, hPSet, fPSet, tTopo} {
  if (HistoMaps_On_) {
    Tk_HM_ = std::make_unique<TkHistoMap>(tkDetMap, "SiStrip/Histo_Map", "LA_TkMap", 0.);
  }
}

SiStripLorentzAngleDQM::~SiStripLorentzAngleDQM() {}

void SiStripLorentzAngleDQM::getActiveDetIds(const edm::EventSetup &eSetup) {
  getConditionObject(eSetup);

  std::map<uint32_t, float>::const_iterator LAMapIter_;
  std::map<uint32_t, float> LAMap_ = condObj_->getLorentzAngles();

  for (LAMapIter_ = LAMap_.begin(); LAMapIter_ != LAMap_.end(); LAMapIter_++) {
    activeDetIds.push_back((*LAMapIter_).first);
  }
}

void SiStripLorentzAngleDQM::fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds) {
  // -----
  // LA on layer-level : fill at once all detIds belonging to same layer when
  // encountering first detID in the layer

  bool fillNext = true;
  for (unsigned int i = 0; i < selectedDetIds.size(); i++) {
    int subDetId_ = DetId(selectedDetIds[i]).subdetId();
    if (subDetId_ < 3 || subDetId_ > 6) {
      edm::LogError("SiStripLorentzAngle")
          << "[SiStripLorentzAngle::fillSummaryMEs] WRONG INPUT : no such "
             "subdetector type : "
          << subDetId_ << " and detId " << selectedDetIds[i] << " therefore no filling!" << std::endl;
    } else if (SummaryOnLayerLevel_On_) {
      if (fillNext) {
        fillMEsForLayer(/*SummaryMEsMap_,*/ selectedDetIds[i]);
      }
      if (getLayerNameAndId(selectedDetIds[i + 1]) == getLayerNameAndId(selectedDetIds[i])) {
        fillNext = false;
      } else {
        fillNext = true;
      }
    } else if (SummaryOnStringLevel_On_) {
      if (fillNext) {
        fillMEsForLayer(/*SummaryMEsMap_,*/ selectedDetIds[i]);
      }
      if (getStringNameAndId(selectedDetIds[i + 1]) == getStringNameAndId(selectedDetIds[i])) {
        fillNext = false;
      } else {
        fillNext = true;
      }
    }
  }

  for (std::map<uint32_t, ModMEs>::iterator iter = SummaryMEsMap_.begin(); iter != SummaryMEsMap_.end(); iter++) {
    ModMEs selME;
    selME = iter->second;

    if (SummaryOnStringLevel_On_) {
      if (fPSet_.getParameter<bool>("OutputSummaryProfileAtLayerLevelAsImage")) {
        TCanvas c1("c1");
        selME.SummaryOfProfileDistr->getTProfile()->Draw();
        std::string name(selME.SummaryOfProfileDistr->getTProfile()->GetTitle());
        name += ".png";
        c1.Print(name.c_str());
      }

      if (fPSet_.getParameter<bool>("OutputCumulativeSummaryAtLayerLevelAsImage")) {
        TCanvas c2("c2");
        selME.SummaryOfCumulDistr->getTH1()->Draw();
        std::string name2(selME.SummaryOfCumulDistr->getTitle());
        name2 += ".png";
        c2.Print(name2.c_str());
      }

    } else {
      if (hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel") &&
          fPSet_.getParameter<bool>("OutputSummaryProfileAtLayerLevelAsImage")) {
        TCanvas c1("c1");
        selME.SummaryOfProfileDistr->getTProfile()->Draw();
        std::string name(selME.SummaryOfProfileDistr->getTProfile()->GetTitle());
        name += ".png";
        c1.Print(name.c_str());
      }

      if (hPSet_.getParameter<bool>("FillCumulativeSummaryAtLayerLevel") &&
          fPSet_.getParameter<bool>("OutputCumulativeSummaryAtLayerLevelAsImage")) {
        TCanvas c1("c1");
        selME.SummaryOfCumulDistr->getTH1()->Draw();
        std::string name(selME.SummaryOfCumulDistr->getTitle());
        name += ".png";
        c1.Print(name.c_str());
      }
    }
  }
}

void SiStripLorentzAngleDQM::fillMEsForLayer(
    /*std::map<uint32_t, ModMEs> selMEsMap_,*/ uint32_t selDetId_) {
  SiStripHistoId hidmanager;

  std::string hSummaryOfProfile_description;
  hSummaryOfProfile_description = hPSet_.getParameter<std::string>("SummaryOfProfile_description");

  std::string hSummary_name;

  int subDetId_ = DetId(selDetId_).subdetId();

  if (subDetId_ < 3 || subDetId_ > 6) {
    edm::LogError("SiStripLorentzAngleDQM") << "[SiStripLorentzAngleDQM::fillMEsForLayer] WRONG INPUT : no such "
                                               "subdetector type : "
                                            << subDetId_ << " no folder set!" << std::endl;
    return;
  }

  std::vector<uint32_t> sameLayerDetIds_;
  sameLayerDetIds_.clear();

  if (SummaryOnStringLevel_On_) {  // FILLING FOR STRING LEVEL

    hSummary_name =
        hidmanager.createHistoLayer(hSummaryOfProfile_description, "layer", getStringNameAndId(selDetId_).first, "");
    std::map<uint32_t, ModMEs>::iterator selMEsMapIter_ = SummaryMEsMap_.find(getStringNameAndId(selDetId_).second);

    ModMEs selME_;
    if (selMEsMapIter_ != SummaryMEsMap_.end())
      selME_ = selMEsMapIter_->second;

    getSummaryMEs(selME_, selDetId_);

    // -----
    sameLayerDetIds_.clear();

    switch (DetId(selDetId_).subdetId()) {
      case StripSubdetector::TIB:
        if (tTopo_->tibIsInternalString(selDetId_)) {
          SiStripSubStructure::getTIBDetectors(
              activeDetIds, sameLayerDetIds_, tTopo_, tTopo_->tibLayer(selDetId_), 0, 1, tTopo_->tibString(selDetId_));
        }
        if (tTopo_->tibIsExternalString(selDetId_)) {
          SiStripSubStructure::getTIBDetectors(
              activeDetIds, sameLayerDetIds_, tTopo_, tTopo_->tibLayer(selDetId_), 0, 2, tTopo_->tibString(selDetId_));
        }
        break;
      case StripSubdetector::TID:
        SiStripSubStructure::getTIDDetectors(activeDetIds, sameLayerDetIds_, tTopo_, 0, 0, 0, 0);
        break;
      case StripSubdetector::TOB:
        SiStripSubStructure::getTOBDetectors(
            activeDetIds, sameLayerDetIds_, tTopo_, tTopo_->tobLayer(selDetId_), 0, tTopo_->tobRod(selDetId_));
        break;
      case StripSubdetector::TEC:
        SiStripSubStructure::getTECDetectors(activeDetIds, sameLayerDetIds_, tTopo_, 0, 0, 0, 0, 0, 0);
        break;
    }

    // -----

    for (unsigned int i = 0; i < sameLayerDetIds_.size(); i++) {
      selME_.SummaryOfProfileDistr->Fill(i + 1, condObj_->getLorentzAngle(sameLayerDetIds_[i]));

      // Fill the Histo_TkMap+TkMap with the LA:
      if (HistoMaps_On_)
        Tk_HM_->fill(sameLayerDetIds_[i], condObj_->getLorentzAngle(sameLayerDetIds_[i]));

      std::cout << sameLayerDetIds_[i] << "\t" << condObj_->getLorentzAngle(sameLayerDetIds_[i]) << std::endl;

      if (fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")) {
        fillTkMap(sameLayerDetIds_[i], condObj_->getLorentzAngle(sameLayerDetIds_[i]));
      }
    }

    std::string hSummaryOfCumul_description;
    hSummaryOfCumul_description = hPSet_.getParameter<std::string>("SummaryOfCumul_description");

    std::string hSummaryOfCumul_name;

    if (subDetId_ < 3 || subDetId_ > 6) {
      edm::LogError("SiStripLorentzAngleDQM") << "[SiStripLorentzAngleDQM::fillMEsForLayer] WRONG INPUT : no such "
                                                 "subdetector type : "
                                              << subDetId_ << " no folder set!" << std::endl;
      return;
    }

    hSummaryOfCumul_name =
        hidmanager.createHistoLayer(hSummaryOfCumul_description, "layer", getStringNameAndId(selDetId_).first, "");

    for (unsigned int i = 0; i < sameLayerDetIds_.size(); i++) {
      selME_.SummaryOfCumulDistr->Fill(condObj_->getLorentzAngle(sameLayerDetIds_[i]));
    }
  }  // FILLING FOR STRING LEVEL

  else {  // FILLING FOR LAYER LEVEL

    std::map<uint32_t, ModMEs>::iterator selMEsMapIter_ = SummaryMEsMap_.find(getLayerNameAndId(selDetId_).second);

    ModMEs selME_;
    if (selMEsMapIter_ != SummaryMEsMap_.end())
      selME_ = selMEsMapIter_->second;

    getSummaryMEs(selME_, selDetId_);

    if (hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel")) {
      hSummary_name =
          hidmanager.createHistoLayer(hSummaryOfProfile_description, "layer", getLayerNameAndId(selDetId_).first, "");

      // -----
      sameLayerDetIds_.clear();

      sameLayerDetIds_ = GetSameLayerDetId(activeDetIds, selDetId_);

      for (unsigned int i = 0; i < sameLayerDetIds_.size(); i++) {
        selME_.SummaryOfProfileDistr->Fill(i + 1, condObj_->getLorentzAngle(sameLayerDetIds_[i]));

        // Fill the Histo_TkMap with LA:
        if (HistoMaps_On_)
          Tk_HM_->fill(sameLayerDetIds_[i], condObj_->getLorentzAngle(sameLayerDetIds_[i]));

        if (fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")) {
          fillTkMap(sameLayerDetIds_[i], condObj_->getLorentzAngle(sameLayerDetIds_[i]));
        }
      }
    }  // if Fill ...

    if (hPSet_.getParameter<bool>("FillCumulativeSummaryAtLayerLevel")) {
      std::string hSummaryOfCumul_description;
      hSummaryOfCumul_description = hPSet_.getParameter<std::string>("SummaryOfCumul_description");

      std::string hSummaryOfCumul_name;

      if (subDetId_ < 3 || subDetId_ > 6) {
        edm::LogError("SiStripLorentzAngleDQM") << "[SiStripLorentzAngleDQM::fillMEsForLayer] WRONG INPUT : no "
                                                   "such subdetector type : "
                                                << subDetId_ << " no folder set!" << std::endl;
        return;
      }

      hSummaryOfCumul_name =
          hidmanager.createHistoLayer(hSummaryOfCumul_description, "layer", getLayerNameAndId(selDetId_).first, "");

      for (unsigned int i = 0; i < sameLayerDetIds_.size(); i++) {
        selME_.SummaryOfCumulDistr->Fill(condObj_->getLorentzAngle(sameLayerDetIds_[i]));
      }
    }  // if Fill ...
  }    // FILLING FOR LAYER LEVEL
}
