#include "../interface/LedClient.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

namespace ecaldqm
{
  LedClient::LedClient() :
    DQWorkerClient(),
    wlToME_(),
    minChannelEntries_(0),
    expectedAmplitude_(0),
    toleranceAmplitude_(0.),
    toleranceAmpRMSRatio_(0.),
    expectedTiming_(0),
    toleranceTiming_(0.),
    toleranceTimRMS_(0.),
    expectedPNAmplitude_(0),
    tolerancePNAmp_(0.),
    tolerancePNRMSRatio_(0.),
    forwardFactor_(0.)
  {
  }

  void
  LedClient::setParams(edm::ParameterSet const& _params)
  {
    minChannelEntries_ = _params.getUntrackedParameter<int>("minChannelEntries");
    toleranceAmplitude_ = _params.getUntrackedParameter<double>("toleranceAmplitude");
    toleranceAmpRMSRatio_ = _params.getUntrackedParameter<double>("toleranceAmpRMSRatio");
    toleranceTiming_ = _params.getUntrackedParameter<double>("toleranceTiming");
    toleranceTimRMS_ = _params.getUntrackedParameter<double>("toleranceTimRMS");
    tolerancePNAmp_ = _params.getUntrackedParameter<double>("tolerancePNAmp");
    tolerancePNRMSRatio_ = _params.getUntrackedParameter<double>("tolerancePNRMSRatio");
    forwardFactor_ = _params.getUntrackedParameter<double>("forwardFactor");

    std::vector<int> ledWavelengths(_params.getUntrackedParameter<std::vector<int> >("ledWavelengths"));

    // wavelengths are not necessarily ordered
    // create a map wl -> MESet index
    // using Amplitude here but any multi-wavelength plot is fine

    MESet::PathReplacements repl;

    MESetMulti const& amplitude(static_cast<MESetMulti const&>(sources_.at("Amplitude")));
    unsigned nWL(ledWavelengths.size());
    for(unsigned iWL(0); iWL != nWL; ++iWL){
      int wl(ledWavelengths[iWL]);
      if(wl != 1 && wl != 2) throw cms::Exception("InvalidConfiguration") << "Led Wavelength";
      repl["wl"] = std::to_string(wl);
      wlToME_[wl] = amplitude.getIndex(repl);
    }

    expectedAmplitude_.resize(nWL);
    expectedTiming_.resize(nWL);
    expectedPNAmplitude_.resize(nWL);

    std::vector<double> inExpectedAmplitude(_params.getUntrackedParameter<std::vector<double> >("expectedAmplitude"));
    std::vector<double> inExpectedTiming(_params.getUntrackedParameter<std::vector<double> >("expectedTiming"));
    std::vector<double> inExpectedPNAmplitude(_params.getUntrackedParameter<std::vector<double> >("expectedPNAmplitude"));

    for(std::map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
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

  void
  LedClient::producePlots(ProcessType)
  {
    uint32_t mask(1 << EcalDQMStatusHelper::LED_MEAN_ERROR |
                  1 << EcalDQMStatusHelper::LED_RMS_ERROR |
                  1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR |
                  1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR);

    MESetMulti& meQuality(static_cast<MESetMulti&>(MEs_.at("Quality")));
    MESetMulti& meQualitySummary(static_cast<MESetMulti&>(MEs_.at("QualitySummary")));
    MESetMulti& meAmplitudeMean(static_cast<MESetMulti&>(MEs_.at("AmplitudeMean")));
    MESetMulti& meAmplitudeRMS(static_cast<MESetMulti&>(MEs_.at("AmplitudeRMS")));
    MESetMulti& meTimingMean(static_cast<MESetMulti&>(MEs_.at("TimingMean")));
    MESetMulti& meTimingRMSMap(static_cast<MESetMulti&>(MEs_.at("TimingRMSMap")));
    MESetMulti& mePNQualitySummary(static_cast<MESetMulti&>(MEs_.at("PNQualitySummary")));

    MESetMulti const& sAmplitude(static_cast<MESetMulti const&>(sources_.at("Amplitude")));
    MESetMulti const& sTiming(static_cast<MESetMulti const&>(sources_.at("Timing")));
    MESetMulti const& sPNAmplitude(static_cast<MESetMulti const&>(sources_.at("PNAmplitude")));

    for(std::map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
      meQuality.use(wlItr->second);
      meQualitySummary.use(wlItr->second);
      meAmplitudeMean.use(wlItr->second);
      meAmplitudeRMS.use(wlItr->second);
      meTimingMean.use(wlItr->second);
      meTimingRMSMap.use(wlItr->second);
      mePNQualitySummary.use(wlItr->second);

      sAmplitude.use(wlItr->second);
      sTiming.use(wlItr->second);
      sPNAmplitude.use(wlItr->second);

      MESet::iterator qEnd(meQuality.end());

      MESet::const_iterator tItr(sTiming);
      MESet::const_iterator aItr(sAmplitude);
      for(MESet::iterator qItr(meQuality.beginChannel()); qItr != qEnd; qItr.toNextChannel()){

        DetId id(qItr->getId());

        bool doMask(meQuality.maskMatches(id, mask, statusManager_));

        aItr = qItr;

        float aEntries(aItr->getBinEntries());

        if(aEntries < minChannelEntries_){
          qItr->setBinContent(doMask ? kMUnknown : kUnknown);
          continue;
        }

        float aMean(aItr->getBinContent());
        float aRms(aItr->getBinError() * sqrt(aEntries));

        meAmplitudeMean.fill(id, aMean);
        meAmplitudeRMS.setBinContent(id, aRms);

        tItr = qItr;

        float tEntries(tItr->getBinEntries());

        if(tEntries < minChannelEntries_) continue;

        float tMean(tItr->getBinContent());
        float tRms(tItr->getBinError() * sqrt(tEntries));

        meTimingMean.fill(id, tMean);
        meTimingRMSMap.setBinContent(id, tRms);

        float intensity(aMean / expectedAmplitude_[wlItr->second]);
        if(isForward(id)) intensity /= forwardFactor_;

        if(intensity < toleranceAmplitude_ || aRms > aMean * toleranceAmpRMSRatio_ ||
           abs(tMean - expectedTiming_[wlItr->second]) > toleranceTiming_ || tRms > toleranceTimRMS_)
          qItr->setBinContent(doMask ? kMBad : kBad);
        else
          qItr->setBinContent(doMask ? kMGood : kGood);
      }

      towerAverage_(meQualitySummary, meQuality, 0.2);

      for(unsigned iDCC(0); iDCC < nDCC; ++iDCC){

        if(memDCCIndex(iDCC + 1) == unsigned(-1)) continue;
        if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;

        for(unsigned iPN(0); iPN < 10; ++iPN){
          EcalPnDiodeDetId id(EcalEndcap, iDCC + 1, iPN + 1);

          bool doMask(mePNQualitySummary.maskMatches(id, mask, statusManager_));

          float pEntries(sPNAmplitude.getBinEntries(id));

          if(pEntries < minChannelEntries_){
            mePNQualitySummary.setBinContent(id, doMask ? kMUnknown : kUnknown);
            continue;
          }

          float pMean(sPNAmplitude.getBinContent(id));
          float pRms(sPNAmplitude.getBinError(id) * sqrt(pEntries));
          float intensity(pMean / expectedPNAmplitude_[wlItr->second]);

          if(intensity < tolerancePNAmp_ || pRms > pMean * tolerancePNRMSRatio_)
            mePNQualitySummary.setBinContent(id, doMask ? kMBad : kBad);
          else
            mePNQualitySummary.setBinContent(id, doMask ? kMGood : kGood);
        }
      }
    }
  }

  DEFINE_ECALDQM_WORKER(LedClient);
}
