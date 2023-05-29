#include "DQM/EcalMonitorClient/interface/LaserClient.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

namespace ecaldqm {
  LaserClient::LaserClient()
      : DQWorkerClient(),
        wlToME_(),
        minChannelEntries_(0),
        expectedAmplitude_(0),
        toleranceAmplitudeLo_(0.),
        toleranceAmplitudeFwdLo_(0.),
        toleranceAmplitudeHi_(0.),
        toleranceAmpRMSRatio_(0.),
        expectedTiming_(0),
        toleranceTiming_(0.),
        toleranceTimRMS_(0.),
        expectedPNAmplitude_(0),
        tolerancePNAmp_(0.),
        tolerancePNRMSRatio_(0.),
        forwardFactor_(0.) {}

  void LaserClient::setParams(edm::ParameterSet const& _params) {
    minChannelEntries_ = _params.getUntrackedParameter<int>("minChannelEntries");
    toleranceAmplitudeLo_ = _params.getUntrackedParameter<double>("toleranceAmplitudeLo");
    toleranceAmplitudeFwdLo_ = _params.getUntrackedParameter<double>("toleranceAmplitudeFwdLo");
    toleranceAmplitudeHi_ = _params.getUntrackedParameter<double>("toleranceAmplitudeHi");
    toleranceAmpRMSRatio_ = _params.getUntrackedParameter<double>("toleranceAmpRMSRatio");
    toleranceTiming_ = _params.getUntrackedParameter<double>("toleranceTiming");
    toleranceTimRMS_ = _params.getUntrackedParameter<double>("toleranceTimRMS");
    tolerancePNAmp_ = _params.getUntrackedParameter<double>("tolerancePNAmp");
    tolerancePNRMSRatio_ = _params.getUntrackedParameter<double>("tolerancePNRMSRatio");
    forwardFactor_ = _params.getUntrackedParameter<double>("forwardFactor");

    std::vector<int> laserWavelengths(_params.getUntrackedParameter<std::vector<int> >("laserWavelengths"));

    // wavelengths are not necessarily ordered
    // create a map wl -> MESet index
    // using Amplitude here but any multi-wavelength plot is fine

    MESet::PathReplacements repl;

    MESetMulti const& amplitude(static_cast<MESetMulti const&>(sources_.at("Amplitude")));
    unsigned nWL(laserWavelengths.size());
    for (unsigned iWL(0); iWL != nWL; ++iWL) {
      int wl(laserWavelengths[iWL]);
      if (wl <= 0 || wl >= 5)
        throw cms::Exception("InvalidConfiguration") << "Laser Wavelength";
      repl["wl"] = std::to_string(wl);
      wlToME_[wl] = amplitude.getIndex(repl);
    }

    expectedAmplitude_.resize(nWL);
    expectedTiming_.resize(nWL);
    expectedPNAmplitude_.resize(nWL);

    std::vector<double> inExpectedAmplitude(_params.getUntrackedParameter<std::vector<double> >("expectedAmplitude"));
    std::vector<double> inExpectedTiming(_params.getUntrackedParameter<std::vector<double> >("expectedTiming"));
    std::vector<double> inExpectedPNAmplitude(
        _params.getUntrackedParameter<std::vector<double> >("expectedPNAmplitude"));

    for (std::map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr) {
      unsigned iME(wlItr->second);
      int iWL(wlItr->first - 1);
      expectedAmplitude_[iME] = inExpectedAmplitude[iWL];
      expectedTiming_[iME] = inExpectedTiming[iWL];
      expectedPNAmplitude_[iME] = inExpectedPNAmplitude[iWL];
    }

    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
    qualitySummaries_.insert("PNQualitySummary");
  }

  void LaserClient::producePlots(ProcessType) {
    uint32_t mask(1 << EcalDQMStatusHelper::LASER_MEAN_ERROR | 1 << EcalDQMStatusHelper::LASER_RMS_ERROR |
                  1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR | 1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR);

    MESetMulti& meQuality(static_cast<MESetMulti&>(MEs_.at("Quality")));
    MESetMulti& meQualitySummary(static_cast<MESetMulti&>(MEs_.at("QualitySummary")));
    MESetMulti& meAmplitudeMean(static_cast<MESetMulti&>(MEs_.at("AmplitudeMean")));
    MESetMulti& meAmplitudeRMS(static_cast<MESetMulti&>(MEs_.at("AmplitudeRMS")));
    MESetMulti& meTimingMean(static_cast<MESetMulti&>(MEs_.at("TimingMean")));
    MESetMulti& meTimingRMSMap(static_cast<MESetMulti&>(MEs_.at("TimingRMSMap")));
    MESetMulti& meTimingRMS(static_cast<MESetMulti&>(MEs_.at("TimingRMS")));
    MESetMulti& mePNQualitySummary(static_cast<MESetMulti&>(MEs_.at("PNQualitySummary")));

    MESetMulti const& sAmplitude(static_cast<MESetMulti const&>(sources_.at("Amplitude")));
    MESetMulti const& sTiming(static_cast<MESetMulti const&>(sources_.at("Timing")));
    MESetMulti const& sPNAmplitude(static_cast<MESetMulti const&>(sources_.at("PNAmplitude")));
    MESet const& sCalibStatus(static_cast<MESet const&>(sources_.at("CalibStatus")));

    for (std::map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr) {
      meQuality.use(wlItr->second);
      meQualitySummary.use(wlItr->second);
      meAmplitudeMean.use(wlItr->second);
      meAmplitudeRMS.use(wlItr->second);
      meTimingMean.use(wlItr->second);
      meTimingRMSMap.use(wlItr->second);
      meTimingRMS.use(wlItr->second);
      mePNQualitySummary.use(wlItr->second);

      sAmplitude.use(wlItr->second);
      sTiming.use(wlItr->second);
      sPNAmplitude.use(wlItr->second);

      MESet::iterator qEnd(meQuality.end(GetElectronicsMap()));

      MESet::const_iterator tItr(GetElectronicsMap(), sTiming);
      MESet::const_iterator aItr(GetElectronicsMap(), sAmplitude);

      int wl(wlItr->first - 1);
      bool enabled(wl < 0 ? false : sCalibStatus.getBinContent(getEcalDQMSetupObjects(), wl) > 0 ? true : false);
      for (MESet::iterator qItr(meQuality.beginChannel(GetElectronicsMap())); qItr != qEnd;
           qItr.toNextChannel(GetElectronicsMap())) {
        DetId id(qItr->getId());

        bool doMask(meQuality.maskMatches(id, mask, statusManager_, GetTrigTowerMap()));

        aItr = qItr;

        float aEntries(aItr->getBinEntries());

        if (aEntries < minChannelEntries_) {
          qItr->setBinContent(enabled ? (doMask ? kMUnknown : kUnknown) : kMUnknown);
          continue;
        }

        float aMean(aItr->getBinContent());
        float aRms(aItr->getBinError() * sqrt(aEntries));

        meAmplitudeMean.fill(getEcalDQMSetupObjects(), id, aMean);
        meAmplitudeRMS.setBinContent(getEcalDQMSetupObjects(), id, aRms);

        tItr = qItr;

        float tEntries(tItr->getBinEntries());

        if (tEntries < minChannelEntries_)
          continue;

        float tMean(tItr->getBinContent());
        float tRms(tItr->getBinError() * sqrt(tEntries));
        float threshAmplitudeLo_;

        meTimingMean.fill(getEcalDQMSetupObjects(), id, tMean);
        meTimingRMS.fill(getEcalDQMSetupObjects(), id, tRms);
        meTimingRMSMap.setBinContent(getEcalDQMSetupObjects(), id, tRms);

        float intensity(aMean / expectedAmplitude_[wlItr->second]);
        if (isForward(id)) {
          intensity /= forwardFactor_;
          threshAmplitudeLo_ = toleranceAmplitudeFwdLo_;
        } else
          threshAmplitudeLo_ = toleranceAmplitudeLo_;

        if (intensity < threshAmplitudeLo_ || intensity > toleranceAmplitudeHi_ ||
            aRms > aMean * toleranceAmpRMSRatio_ ||
            std::abs(tMean - expectedTiming_[wlItr->second]) > toleranceTiming_ /*|| tRms > toleranceTimRMS_*/)
          qItr->setBinContent(doMask ? kMBad : kBad);
        else
          qItr->setBinContent(doMask ? kMGood : kGood);
      }

      towerAverage_(meQualitySummary, meQuality, 0.2);

      for (unsigned iDCC(0); iDCC < nDCC; ++iDCC) {
        if (memDCCIndex(iDCC + 1) == unsigned(-1))
          continue;
        int subdet(0);
        if (iDCC >= kEBmLow && iDCC <= kEBpHigh)
          subdet = EcalBarrel;
        else
          subdet = EcalEndcap;

        for (unsigned iPN(0); iPN < 10; ++iPN) {
          EcalPnDiodeDetId id(subdet, iDCC + 1, iPN + 1);

          bool doMask(mePNQualitySummary.maskMatches(id, mask, statusManager_, GetTrigTowerMap()));

          float pEntries(sPNAmplitude.getBinEntries(getEcalDQMSetupObjects(), id));

          if (pEntries < minChannelEntries_) {
            mePNQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMUnknown : kUnknown);
            continue;
          }

          float pMean(sPNAmplitude.getBinContent(getEcalDQMSetupObjects(), id));
          float pRms(sPNAmplitude.getBinError(getEcalDQMSetupObjects(), id) * sqrt(pEntries));
          float intensity(pMean / expectedPNAmplitude_[wlItr->second]);

          if (intensity < tolerancePNAmp_ || pRms > pMean * tolerancePNRMSRatio_)
            mePNQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMBad : kBad);
          else
            mePNQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMGood : kGood);
        }
      }
    }
  }

  DEFINE_ECALDQM_WORKER(LaserClient);
}  // namespace ecaldqm
