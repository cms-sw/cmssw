#include "DQM/EcalMonitorClient/interface/TowerStatusTask.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

namespace ecaldqm {

  TowerStatusTask::TowerStatusTask() : DQWorkerClient(), doDAQInfo_(false), doDCSInfo_(false) {
    std::fill_n(daqStatus_, nDCC, 0.);
    std::fill_n(dcsStatus_, nDCC, 0.);
  }

  void TowerStatusTask::setParams(edm::ParameterSet const& _params) {
    doDAQInfo_ = _params.getUntrackedParameter<bool>("doDAQInfo");
    doDCSInfo_ = _params.getUntrackedParameter<bool>("doDCSInfo");

    if (doDAQInfo_ && doDCSInfo_)
      return;
    if (doDAQInfo_) {
      MEs_.erase(std::string("DCSSummary"));
      MEs_.erase(std::string("DCSSummaryMap"));
      MEs_.erase(std::string("DCSContents"));
    } else if (doDCSInfo_) {
      MEs_.erase(std::string("DAQSummary"));
      MEs_.erase(std::string("DAQSummaryMap"));
      MEs_.erase(std::string("DAQContents"));
    } else
      throw cms::Exception("InvalidConfiguration") << "Nothing to do in TowerStatusTask";
  }

  void TowerStatusTask::setTokens(edm::ConsumesCollector& _collector) {
    daqHndlToken = _collector.esConsumes<edm::Transition::EndLuminosityBlock>();
    dcsHndlToken = _collector.esConsumes<edm::Transition::EndLuminosityBlock>();
  }

  void TowerStatusTask::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& _es) {
    if (doDAQInfo_) {
      std::fill_n(daqStatus_, nDCC, 1.);

      const EcalDAQTowerStatus* daqHndl = &_es.getData(daqHndlToken);
      auto daqhandle = _es.getHandle(daqHndlToken);
      if (daqhandle.isValid()) {
        for (unsigned id(0); id < EcalTrigTowerDetId::kEBTotalTowers; id++) {
          if (daqHndl->barrel(id).getStatusCode() != 0) {
            EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(id));
            daqStatus_[dccId(ttid, GetElectronicsMap()) - 1] -= 25. / 1700.;
          }
        }
        for (unsigned id(0); id < EcalScDetId::kSizeForDenseIndexing; id++) {
          if (daqHndl->endcap(id).getStatusCode() != 0) {
            EcalScDetId scid(EcalScDetId::unhashIndex(id));
            unsigned dccid(dccId(scid, GetElectronicsMap()));
            daqStatus_[dccid - 1] -= double(scConstituents(scid).size()) / nCrystals(dccid);
          }
        }
      } else
        edm::LogWarning("EventSetup") << "EcalDAQTowerStatus record not valid";
    }

    if (doDCSInfo_) {
      std::fill_n(dcsStatus_, nDCC, 1.);

      const EcalDCSTowerStatus* dcsHndl = &_es.getData(dcsHndlToken);
      auto dcshandle = _es.getHandle(dcsHndlToken);
      if (dcshandle.isValid()) {
        for (unsigned id(0); id < EcalTrigTowerDetId::kEBTotalTowers; id++) {
          if (dcsHndl->barrel(id).getStatusCode() != 0) {
            EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(id));
            dcsStatus_[dccId(ttid, GetElectronicsMap()) - 1] -= 25. / 1700.;
          }
        }
        for (unsigned id(0); id < EcalScDetId::kSizeForDenseIndexing; id++) {
          if (dcsHndl->endcap(id).getStatusCode() != 0) {
            EcalScDetId scid(EcalScDetId::unhashIndex(id));
            unsigned dccid(dccId(scid, GetElectronicsMap()));
            dcsStatus_[dccid - 1] -= double(scConstituents(scid).size()) / nCrystals(dccid);
          }
        }
      } else
        edm::LogWarning("EventSetup") << "EcalDCSTowerStatus record not valid";
    }
  }

  void TowerStatusTask::producePlots(ProcessType) {
    if (doDAQInfo_)
      producePlotsTask_(daqStatus_, "DAQ");
    if (doDCSInfo_)
      producePlotsTask_(dcsStatus_, "DCS");
  }

  void TowerStatusTask::producePlotsTask_(float const* _status, std::string const& _type) {
    MESet* meSummary(nullptr);
    MESet* meSummaryMap(nullptr);
    MESet* meContents(nullptr);
    meSummary = &MEs_.at(_type + "Summary");
    meSummaryMap = &MEs_.at(_type + "SummaryMap");
    meContents = &MEs_.at(_type + "Contents");

    meSummary->reset(GetElectronicsMap(), -1.);
    meSummaryMap->resetAll(-1.);
    meSummaryMap->reset(GetElectronicsMap());
    meContents->reset(GetElectronicsMap(), -1.);

    float totalFraction(0.);
    for (int iDCC(0); iDCC < nDCC; iDCC++) {
      meSummaryMap->setBinContent(getEcalDQMSetupObjects(), iDCC + 1, _status[iDCC]);
      meContents->fill(getEcalDQMSetupObjects(), iDCC + 1, _status[iDCC]);
      totalFraction += _status[iDCC] / nCrystals(iDCC + 1);
    }

    meSummary->fill(getEcalDQMSetupObjects(), totalFraction);
  }

  DEFINE_ECALDQM_WORKER(TowerStatusTask);
}  // namespace ecaldqm
