#include "DQM/EcalMonitorClient/interface/PresampleClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

namespace ecaldqm {
  PresampleClient::PresampleClient()
      : DQWorkerClient(),
        minChannelEntries_(0),
        expectedMean_(0.),
        toleranceLow_(0.),
        toleranceHigh_(0.),
        toleranceRMS_(0.),
        toleranceRMSFwd_(0.) {
    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
  }

  void PresampleClient::setParams(edm::ParameterSet const& _params) {
    minChannelEntries_ = _params.getUntrackedParameter<int>("minChannelEntries");
    expectedMean_ = _params.getUntrackedParameter<double>("expectedMean");
    toleranceLow_ = _params.getUntrackedParameter<double>("toleranceLow");
    toleranceHigh_ = _params.getUntrackedParameter<double>("toleranceHigh");
    toleranceRMS_ = _params.getUntrackedParameter<double>("toleranceRMS");
    toleranceRMSFwd_ = _params.getUntrackedParameter<double>("toleranceRMSFwd");
  }

  void PresampleClient::producePlots(ProcessType) {
    MESet& meQualitySummary(MEs_.at("QualitySummary"));
    MESet& meQuality(MEs_.at("Quality"));
    MESet& meErrorsSummary(MEs_.at("ErrorsSummary"));
    MESet& meMean(MEs_.at("Mean"));
    MESet& meRMS(MEs_.at("RMS"));
    MESet& meRMSMap(MEs_.at("RMSMap"));
    MESet& meRMSMapAll(MEs_.at("RMSMapAll"));
    MESet& meRMSMapAllByLumi(MEs_.at("RMSMapAllByLumi"));
    MESet& meMeanMapAll(MEs_.at("MeanMapAll"));

    MESet const& sPedestal(sources_.at("Pedestal"));
    MESet const& sPedestalByLS(sources_.at("PedestalByLS"));
    MESet const& sChStatus(sources_.at("ChStatus"));

    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
                  1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR);

    MESet::iterator qEnd(meQuality.end(GetElectronicsMap()));

    MESet::const_iterator pItr(GetElectronicsMap(), sPedestal);
    MESet::const_iterator pLSItr(GetElectronicsMap(), sPedestalByLS);
    double maxEB(0.), minEB(0.), maxEE(0.), minEE(0.);
    double rmsMaxEB(0.), rmsMaxEE(0.);
    for (MESet::iterator qItr(meQuality.beginChannel(GetElectronicsMap())); qItr != qEnd;
         qItr.toNextChannel(GetElectronicsMap())) {
      pItr = qItr;
      pLSItr = qItr;

      DetId id(qItr->getId());

      bool doMask(meQuality.maskMatches(id, mask, statusManager_, GetTrigTowerMap()));

      double rmsThresh(toleranceRMS_);

      if (isForward(id))
        rmsThresh = toleranceRMSFwd_;

      double entries(pItr->getBinEntries());
      double entriesLS(pLSItr->getBinEntries());

      if (entries < minChannelEntries_) {
        qItr->setBinContent(doMask ? kMUnknown : kUnknown);
        meQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMUnknown : kUnknown);
        meRMSMap.setBinContent(getEcalDQMSetupObjects(), id, -1.);
        continue;
      }

      double mean(pItr->getBinContent());
      double meanLS(pLSItr->getBinContent());
      double rms(pItr->getBinError() * std::sqrt(entries));
      double rmsLS(pLSItr->getBinError() * std::sqrt(entriesLS));

      int dccid(dccId(id, GetElectronicsMap()));

      meMean.fill(getEcalDQMSetupObjects(), dccid, mean);
      meRMS.fill(getEcalDQMSetupObjects(), dccid, rms);
      meRMSMap.setBinContent(getEcalDQMSetupObjects(), id, rms);
      meRMSMapAllByLumi.setBinContent(getEcalDQMSetupObjects(), id, rmsLS);

      if (((mean > expectedMean_ + toleranceHigh_) || (mean < expectedMean_ - toleranceLow_)) || rms > rmsThresh) {
        qItr->setBinContent(doMask ? kMBad : kBad);
        meQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMBad : kBad);
        if (!doMask)
          meErrorsSummary.fill(getEcalDQMSetupObjects(), id);
      } else {
        qItr->setBinContent(doMask ? kMGood : kGood);
        meQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMGood : kGood);
      }

      // Fill Presample Trend plots:
      // Use PedestalByLS which only contains digis from "current" LS
      float chStatus(sChStatus.getBinContent(getEcalDQMSetupObjects(), id));
      if (entriesLS < minChannelEntries_)
        continue;
      if (chStatus != EcalChannelStatusCode::kOk)
        continue;  // exclude problematic channels

      // Get max/min
      // Min is effectively just 0
      if (id.subdetId() == EcalBarrel) {
        if (meanLS > maxEB)
          maxEB = meanLS;
        if (meanLS < minEB)
          minEB = meanLS;
        if (rmsLS > rmsMaxEB)
          rmsMaxEB = rmsLS;
      } else {
        if (meanLS > maxEE)
          maxEE = meanLS;
        if (meanLS < minEE)
          minEE = meanLS;
        if (rmsLS > rmsMaxEE)
          rmsMaxEE = rmsLS;
      }

    }  // qItr

    towerAverage_(meRMSMapAll, meRMSMap, -1.);
    towerAverage_(meMeanMapAll, sPedestal, -1.);

    MESet& meTrendMean(MEs_.at("TrendMean"));
    MESet& meTrendRMS(MEs_.at("TrendRMS"));
    meTrendMean.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), maxEB - minEB);
    meTrendMean.fill(getEcalDQMSetupObjects(), EcalEndcap, double(timestamp_.iLumi), maxEE - minEE);
    meTrendRMS.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), rmsMaxEB);
    meTrendRMS.fill(getEcalDQMSetupObjects(), EcalEndcap, double(timestamp_.iLumi), rmsMaxEE);
  }

  DEFINE_ECALDQM_WORKER(PresampleClient);
}  // namespace ecaldqm
