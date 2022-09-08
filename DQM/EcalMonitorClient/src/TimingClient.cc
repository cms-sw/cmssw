#include "DQM/EcalMonitorClient/interface/TimingClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

namespace ecaldqm {
  TimingClient::TimingClient()
      : DQWorkerClient(),
        ebtoleranceMean_(0.),
        eetoleranceMean_(0.),
        toleranceMeanFwd_(0.),
        toleranceRMS_(0.),
        toleranceRMSFwd_(0.),
        minChannelEntries_(0),
        minChannelEntriesFwd_(0),
        minTowerEntries_(0),
        minTowerEntriesFwd_(0),
        tailPopulThreshold_(0.) {
    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
  }

  void TimingClient::setParams(edm::ParameterSet const& _params) {
    ebtoleranceMean_ = _params.getUntrackedParameter<double>("ebtoleranceMean");
    eetoleranceMean_ = _params.getUntrackedParameter<double>("eetoleranceMean");
    toleranceMeanFwd_ = _params.getUntrackedParameter<double>("toleranceMeanFwd");
    toleranceRMS_ = _params.getUntrackedParameter<double>("toleranceRMS");
    toleranceRMSFwd_ = _params.getUntrackedParameter<double>("toleranceRMSFwd");
    minChannelEntries_ = _params.getUntrackedParameter<int>("minChannelEntries");
    minChannelEntriesFwd_ = _params.getUntrackedParameter<int>("minChannelEntriesFwd");
    minTowerEntries_ = _params.getUntrackedParameter<int>("minTowerEntries");
    minTowerEntriesFwd_ = _params.getUntrackedParameter<int>("minChannelEntriesFwd");
    tailPopulThreshold_ = _params.getUntrackedParameter<double>("tailPopulThreshold");
  }

  void TimingClient::producePlots(ProcessType) {
    MESet& meQuality(MEs_.at("Quality"));
    MESet& meMeanSM(MEs_.at("MeanSM"));
    MESet& meMeanAll(MEs_.at("MeanAll"));
    MESet& meFwdBkwdDiff(MEs_.at("FwdBkwdDiff"));
    MESet& meFwdvBkwd(MEs_.at("FwdvBkwd"));
    MESet& meRMSMap(MEs_.at("RMSMap"));
    MESet& meRMSAll(MEs_.at("RMSAll"));
    MESet& meProjEta(MEs_.at("ProjEta"));
    MESet& meProjPhi(MEs_.at("ProjPhi"));
    MESet& meQualitySummary(MEs_.at("QualitySummary"));

    MESet const& sTimeAllMap(sources_.at("TimeAllMap"));
    MESet const& sTimeMap(sources_.at("TimeMap"));
    MESet const& sTimeMapByLS(sources_.at("TimeMapByLS"));
    MESet const& sChStatus(sources_.at("ChStatus"));

    uint32_t mask(1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING);

    MESet::iterator qEnd(meQuality.end(GetElectronicsMap()));

    MESet::iterator rItr(GetElectronicsMap(), meRMSMap);
    MESet::const_iterator tItr(GetElectronicsMap(), sTimeMap);
    MESet::const_iterator tLSItr(GetElectronicsMap(), sTimeMapByLS);

    float EBentries(0.), EEentries(0.);
    float EBmean(0.), EEmean(0.);
    float EBrms(0.), EErms(0.);
    for (MESet::iterator qItr(meQuality.beginChannel(GetElectronicsMap())); qItr != qEnd;
         qItr.toNextChannel(GetElectronicsMap())) {
      tItr = qItr;
      rItr = qItr;

      DetId id(qItr->getId());

      int minChannelEntries(minChannelEntries_);
      float meanThresh;
      float rmsThresh(toleranceRMS_);

      if (id.subdetId() == EcalBarrel)
        meanThresh = ebtoleranceMean_;
      else
        meanThresh = eetoleranceMean_;

      if (isForward(id)) {
        minChannelEntries = minChannelEntriesFwd_;
        meanThresh = toleranceMeanFwd_;
        rmsThresh = toleranceRMSFwd_;
      }

      bool doMask(meQuality.maskMatches(id, mask, statusManager_, GetTrigTowerMap()));

      float entries(tItr->getBinEntries());

      if (entries < minChannelEntries) {
        qItr->setBinContent(doMask ? kMUnknown : kUnknown);
        rItr->setBinContent(-1.);
        continue;
      }

      float mean(tItr->getBinContent());
      float rms(tItr->getBinError() * sqrt(entries));

      meMeanSM.fill(getEcalDQMSetupObjects(), id, mean);
      meMeanAll.fill(getEcalDQMSetupObjects(), id, mean);
      meProjEta.fill(getEcalDQMSetupObjects(), id, mean);
      meProjPhi.fill(getEcalDQMSetupObjects(), id, mean);
      meRMSAll.fill(getEcalDQMSetupObjects(), id, rms);
      rItr->setBinContent(rms);

      bool negative(false);
      float posTime(0.);

      if (id.subdetId() == EcalBarrel) {
        EBDetId ebid(id);
        if (ebid.zside() < 0) {
          negative = true;
          EBDetId posId(EBDetId::switchZSide(ebid));
          posTime = sTimeMap.getBinContent(getEcalDQMSetupObjects(), posId);
        }
      } else {
        EEDetId eeid(id);
        if (eeid.zside() < 0) {
          negative = true;
          EEDetId posId(EEDetId::switchZSide(eeid));
          posTime = sTimeMap.getBinContent(getEcalDQMSetupObjects(), posId);
        }
      }
      if (negative) {
        meFwdBkwdDiff.fill(getEcalDQMSetupObjects(), id, posTime - mean);
        meFwdvBkwd.fill(getEcalDQMSetupObjects(), id, mean, posTime);
      }

      if (std::abs(mean) > meanThresh || rms > rmsThresh)
        qItr->setBinContent(doMask ? kMBad : kBad);
      else
        qItr->setBinContent(doMask ? kMGood : kGood);

      // For Trend plots:
      tLSItr = qItr;
      float entriesLS(tLSItr->getBinEntries());
      float meanLS(tLSItr->getBinContent());
      float rmsLS(tLSItr->getBinError() * sqrt(entriesLS));
      float chStatus(sChStatus.getBinContent(getEcalDQMSetupObjects(), id));

      if (entriesLS < minChannelEntries)
        continue;
      if (chStatus != EcalChannelStatusCode::kOk)
        continue;  // exclude problematic channels

      // Keep running count of timing mean, rms, and N_hits
      if (id.subdetId() == EcalBarrel) {
        EBmean += meanLS;
        EBrms += rmsLS;
        EBentries += entriesLS;
      } else {
        EEmean += meanLS;
        EErms += rmsLS;
        EEentries += entriesLS;
      }

    }  // channel loop

    // Fill Timing Trend plots at each LS
    MESet& meTrendMean(MEs_.at("TrendMean"));
    MESet& meTrendRMS(MEs_.at("TrendRMS"));
    if (EBentries > 0.) {
      if (std::abs(EBmean) > 0.)
        meTrendMean.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), EBmean / EBentries);
      if (std::abs(EBrms) > 0.)
        meTrendRMS.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), EBrms / EBentries);
    }
    if (EEentries > 0.) {
      if (std::abs(EEmean) > 0.)
        meTrendMean.fill(getEcalDQMSetupObjects(), EcalEndcap, double(timestamp_.iLumi), EEmean / EEentries);
      if (std::abs(EErms) > 0.)
        meTrendRMS.fill(getEcalDQMSetupObjects(), EcalEndcap, double(timestamp_.iLumi), EErms / EEentries);
    }

    MESet::iterator qsEnd(meQualitySummary.end(GetElectronicsMap()));

    for (MESet::iterator qsItr(meQualitySummary.beginChannel(GetElectronicsMap())); qsItr != qsEnd;
         qsItr.toNextChannel(GetElectronicsMap())) {
      DetId tId(qsItr->getId());

      std::vector<DetId> ids;

      if (tId.subdetId() == EcalTriggerTower)
        ids = GetTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(tId));
      else
        ids = scConstituents(EcalScDetId(tId));

      int minTowerEntries(minTowerEntries_);
      float meanThresh;
      float rmsThresh(toleranceRMS_);

      if (tId.subdetId() == EcalBarrel)
        meanThresh = ebtoleranceMean_;
      else
        meanThresh = eetoleranceMean_;

      if (isForward(tId)) {
        minTowerEntries = minTowerEntriesFwd_;
        meanThresh = toleranceMeanFwd_;
        rmsThresh = toleranceRMSFwd_;
      }

      // tower entries != sum(channel entries) because of the difference in timing cut at the source
      float summaryEntries(sTimeAllMap.getBinEntries(getEcalDQMSetupObjects(), tId));

      float towerEntries(0.);
      float towerMean(0.);
      float towerMean2(0.);

      bool doMask(false);

      for (std::vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr) {
        DetId& id(*idItr);

        doMask |= meQuality.maskMatches(id, mask, statusManager_, GetTrigTowerMap());

        MESet::const_iterator tmItr(GetElectronicsMap(), sTimeMap, id);

        float entries(tmItr->getBinEntries());
        if (entries < 0.)
          continue;
        towerEntries += entries;
        float mean(tmItr->getBinContent());
        towerMean += mean * entries;
        float rms(tmItr->getBinError() * sqrt(entries));
        towerMean2 += (rms * rms + mean * mean) * entries;
      }

      double quality(doMask ? kMUnknown : kUnknown);
      if (towerEntries / ids.size() > minTowerEntries / 25.) {
        if (summaryEntries < towerEntries * (1. - tailPopulThreshold_))  // large timing deviation
          quality = doMask ? kMBad : kBad;
        else {
          towerMean /= towerEntries;
          towerMean2 /= towerEntries;

          float towerRMS(sqrt(towerMean2 - towerMean * towerMean));

          if (std::abs(towerMean) > meanThresh || towerRMS > rmsThresh)
            quality = doMask ? kMBad : kBad;
          else
            quality = doMask ? kMGood : kGood;
        }
      }
      qsItr->setBinContent(quality);
    }
  }

  DEFINE_ECALDQM_WORKER(TimingClient);
}  // namespace ecaldqm
