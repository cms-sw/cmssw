#include "../interface/LaserClient.h"
#include "../interface/EcalDQMClientUtils.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

#include <cmath>

namespace ecaldqm {

  LaserClient::LaserClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "LaserClient"),
    wlToME_(),
    minChannelEntries_(_workerParams.getUntrackedParameter<int>("minChannelEntries")),
    expectedAmplitude_(0),
    toleranceAmplitude_(_workerParams.getUntrackedParameter<double>("toleranceAmplitude")),
    toleranceAmpRMSRatio_(_workerParams.getUntrackedParameter<double>("toleranceAmpRMSRatio")),
    expectedTiming_(0),
    toleranceTiming_(_workerParams.getUntrackedParameter<double>("toleranceTiming")),
    toleranceTimRMS_(_workerParams.getUntrackedParameter<double>("toleranceTimRMS")),
    expectedPNAmplitude_(0),
    tolerancePNAmp_(_workerParams.getUntrackedParameter<double>("tolerancePNAmp")),
    tolerancePNRMSRatio_(_workerParams.getUntrackedParameter<double>("tolerancePNRMSRatio")),
    forwardFactor_(_workerParams.getUntrackedParameter<double>("forwardFactor"))
  {
    using namespace std;

    vector<int> laserWavelengths(_commonParams.getUntrackedParameter<vector<int> >("laserWavelengths"));

    unsigned iMEWL(0);
    for(vector<int>::iterator wlItr(laserWavelengths.begin()); wlItr != laserWavelengths.end(); ++wlItr){
      if(*wlItr <= 0 || *wlItr >= 5) throw cms::Exception("InvalidConfiguration") << "Laser Wavelength" << endl;
      wlToME_[*wlItr] = iMEWL++;
    }

    expectedAmplitude_.resize(iMEWL);
    expectedTiming_.resize(iMEWL);
    expectedPNAmplitude_.resize(iMEWL);

    std::vector<double> inExpectedAmplitude(_workerParams.getUntrackedParameter<std::vector<double> >("expectedAmplitude"));
    std::vector<double> inExpectedTiming(_workerParams.getUntrackedParameter<std::vector<double> >("expectedTiming"));
    std::vector<double> inExpectedPNAmplitude(_workerParams.getUntrackedParameter<std::vector<double> >("expectedPNAmplitude"));

    for(map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
      int iME(wlItr->second);
      int iWL(wlItr->first - 1);
      expectedAmplitude_[iME] = inExpectedAmplitude[iWL];
      expectedTiming_[iME] = inExpectedTiming[iWL];
      expectedPNAmplitude_[iME] = inExpectedPNAmplitude[iWL];
    }

    map<string, string> replacements;
    stringstream ss;

    std::string wlPlots[] = {"Quality", "AmplitudeMean", "AmplitudeRMS", "TimingMean", "TimingRMSMap", "TimingRMS", "QualitySummary", "PNQualitySummary"};
    for(unsigned iS(0); iS < sizeof(wlPlots) / sizeof(std::string); ++iS){
      std::string& plot(wlPlots[iS]);
      MESetMulti* multi(static_cast<MESetMulti*>(MEs_[plot]));

      for(map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
        multi->use(wlItr->second);

        ss.str("");
        ss << wlItr->first;
        replacements["wl"] = ss.str();

        multi->formPath(replacements);
      }
    }

    std::string wlSources[] = {"Amplitude", "Timing", "PNAmplitude"};
    for(unsigned iS(0); iS < sizeof(wlSources) / sizeof(std::string); ++iS){
      std::string& plot(wlSources[iS]);
      MESetMulti const* multi(static_cast<MESetMulti const*>(sources_[plot]));

      for(map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
        multi->use(wlItr->second);

        ss.str("");
        ss << wlItr->first;
        replacements["wl"] = ss.str();

        multi->formPath(replacements);
      }
    }

    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
    qualitySummaries_.insert("PNQualitySummary");
  }

  void
  LaserClient::producePlots()
  {
    uint32_t mask(1 << EcalDQMStatusHelper::LASER_MEAN_ERROR |
                  1 << EcalDQMStatusHelper::LASER_RMS_ERROR |
                  1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR |
                  1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR);

    MESetMulti* meQuality(static_cast<MESetMulti*>(MEs_["Quality"]));
    MESetMulti* meQualitySummary(static_cast<MESetMulti*>(MEs_["QualitySummary"]));
    MESetMulti* meAmplitudeMean(static_cast<MESetMulti*>(MEs_["AmplitudeMean"]));
    MESetMulti* meAmplitudeRMS(static_cast<MESetMulti*>(MEs_["AmplitudeRMS"]));
    MESetMulti* meTimingMean(static_cast<MESetMulti*>(MEs_["TimingMean"]));
    MESetMulti* meTimingRMSMap(static_cast<MESetMulti*>(MEs_["TimingRMSMap"]));
    MESetMulti* meTimingRMS(static_cast<MESetMulti*>(MEs_["TimingRMS"]));
    MESetMulti* mePNQualitySummary(static_cast<MESetMulti*>(MEs_["PNQualitySummary"]));

    MESetMulti const* sAmplitude(static_cast<MESetMulti const*>(sources_["Amplitude"]));
    MESetMulti const* sTiming(static_cast<MESetMulti const*>(sources_["Timing"]));
    MESetMulti const* sPNAmplitude(static_cast<MESetMulti const*>(sources_["PNAmplitude"]));

    for(std::map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
      meQuality->use(wlItr->second);
      meQualitySummary->use(wlItr->second);
      meAmplitudeMean->use(wlItr->second);
      meAmplitudeRMS->use(wlItr->second);
      meTimingMean->use(wlItr->second);
      meTimingRMSMap->use(wlItr->second);
      meTimingRMS->use(wlItr->second);
      mePNQualitySummary->use(wlItr->second);

      sAmplitude->use(wlItr->second);
      sTiming->use(wlItr->second);
      sPNAmplitude->use(wlItr->second);

      meAmplitudeMean->reset();
      meAmplitudeRMS->reset();
      meTimingMean->reset();
      meTimingRMSMap->reset();
      meTimingRMS->reset();

      MESet::iterator qEnd(meQuality->end());

      MESet::const_iterator tItr(sTiming);
      MESet::const_iterator aItr(sAmplitude);
      for(MESet::iterator qItr(meQuality->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

        DetId id(qItr->getId());

        bool doMask(applyMask(meQuality->getBinType(), id, mask));

        aItr = qItr;

        float aEntries(aItr->getBinEntries());

        if(aEntries < minChannelEntries_){
          qItr->setBinContent(doMask ? kMUnknown : kUnknown);
          continue;
        }

        float aMean(aItr->getBinContent());
        float aRms(aItr->getBinError() * sqrt(aEntries));

        meAmplitudeMean->fill(id, aMean);
        meAmplitudeRMS->setBinContent(id, aRms);

        tItr = qItr;

        float tEntries(tItr->getBinEntries());

        if(tEntries < minChannelEntries_) continue;

        float tMean(tItr->getBinContent());
        float tRms(tItr->getBinError() * sqrt(tEntries));

        meTimingMean->fill(id, tMean);
        meTimingRMS->fill(id, tRms);
        meTimingRMSMap->setBinContent(id, tRms);

        float intensity(aMean / expectedAmplitude_[wlItr->second]);
        if(isForward(id)) intensity /= forwardFactor_;

        if(intensity < toleranceAmplitude_ || aRms > aMean * toleranceAmpRMSRatio_ ||
           abs(tMean - expectedTiming_[wlItr->second]) > toleranceTiming_ /*|| tRms > toleranceTimRMS_*/)
          qItr->setBinContent(doMask ? kMBad : kBad);
        else
          qItr->setBinContent(doMask ? kMGood : kGood);
      }

      towerAverage_(meQualitySummary, meQuality, 0.2);

      for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){

        if(memDCCIndex(iDCC + 1) == unsigned(-1)) continue;
        int subdet(0);
        if(iDCC >= kEBmLow && iDCC <= kEBpHigh) subdet = EcalBarrel;
        else subdet = EcalEndcap;

        for(unsigned iPN(0); iPN < 10; ++iPN){
          EcalPnDiodeDetId id(subdet, iDCC + 1, iPN + 1);

          bool doMask(applyMask(mePNQualitySummary->getBinType(), id, mask));

          float pEntries(sPNAmplitude->getBinEntries(id));

          if(pEntries < minChannelEntries_){
            mePNQualitySummary->setBinContent(id, doMask ? kMUnknown : kUnknown);
            continue;
          }

          float pMean(sPNAmplitude->getBinContent(id));
          float pRms(sPNAmplitude->getBinError(id) * sqrt(pEntries));
          float intensity(pMean / expectedPNAmplitude_[wlItr->second]);

          if(intensity < tolerancePNAmp_ || pRms > pMean * tolerancePNRMSRatio_)
            mePNQualitySummary->setBinContent(id, doMask ? kMBad : kBad);
          else
            mePNQualitySummary->setBinContent(id, doMask ? kMGood : kGood);
        }
      }
    }
  }

  DEFINE_ECALDQM_WORKER(LaserClient);
}
