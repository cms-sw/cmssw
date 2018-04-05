#include "../interface/TestPulseClient.h"

#include "DQM/EcalCommon/interface/MESetMulti.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iomanip>

namespace ecaldqm
{
  TestPulseClient::TestPulseClient() :
    DQWorkerClient(),
    gainToME_(),
    pnGainToME_(),
    minChannelEntries_(0),
    amplitudeThreshold_(0),
    toleranceRMS_(0),
    PNAmplitudeThreshold_(0),
    tolerancePNRMS_(0)
  {
  }

  void
  TestPulseClient::setParams(edm::ParameterSet const& _params)
  {
    minChannelEntries_ = _params.getUntrackedParameter<int>("minChannelEntries");

    std::vector<int> MGPAGains(_params.getUntrackedParameter<std::vector<int> >("MGPAGains"));
    std::vector<int> MGPAGainsPN(_params.getUntrackedParameter<std::vector<int> >("MGPAGainsPN"));

    MESet::PathReplacements repl;

    MESetMulti const& amplitude(static_cast<MESetMulti const&>(sources_.at("Amplitude")));
    unsigned nG(MGPAGains.size());
    for(unsigned iG(0); iG != nG; ++iG){
      int gain(MGPAGains[iG]);
      if(gain != 1 && gain != 6 && gain != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain";
      repl["gain"] = std::to_string(gain);
      gainToME_[gain] = amplitude.getIndex(repl);
    }

    repl.clear();

    MESetMulti const& pnAmplitude(static_cast<MESetMulti const&>(sources_.at("PNAmplitude")));
    unsigned nGPN(MGPAGainsPN.size());
    for(unsigned iG(0); iG != nGPN; ++iG){
      int gain(MGPAGainsPN[iG]);
      if(gain != 1 && gain != 16) throw cms::Exception("InvalidConfiguration") << "PN MGPA gain";
      repl["pngain"] = std::to_string(gain);
      pnGainToME_[gain] = pnAmplitude.getIndex(repl);
    }

    amplitudeThreshold_.resize(nG);
    toleranceRMS_.resize(nG);

    std::vector<double> inAmplitudeThreshold(_params.getUntrackedParameter<std::vector<double> >("amplitudeThreshold"));
    std::vector<double> inToleranceRMS(_params.getUntrackedParameter<std::vector<double> >("toleranceRMS"));

    for(std::map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      unsigned iME(gainItr->second);
      unsigned iGain(0);
      switch(gainItr->first){
      case 1:
        iGain = 0; break;
      case 6:
        iGain = 1; break;
      case 12:
        iGain = 2; break;
      }

      amplitudeThreshold_[iME] = inAmplitudeThreshold[iGain];
      toleranceRMS_[iME] = inToleranceRMS[iGain];
    }

    PNAmplitudeThreshold_.resize(nGPN);
    tolerancePNRMS_.resize(nGPN);

    std::vector<double> inPNAmplitudeThreshold(_params.getUntrackedParameter<std::vector<double> >("PNAmplitudeThreshold"));
    std::vector<double> inTolerancePNRMS(_params.getUntrackedParameter<std::vector<double> >("tolerancePNRMS"));

    for(std::map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      unsigned iME(gainItr->second);
      unsigned iGain(0);
      switch(gainItr->first){
      case 1:
        iGain = 0; break;
      case 16:
        iGain = 1; break;
      }

      PNAmplitudeThreshold_[iME] = inPNAmplitudeThreshold[iGain];
      tolerancePNRMS_[iME] = inTolerancePNRMS[iGain];
    }

    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
    qualitySummaries_.insert("PNQualitySummary");
  }

  void
  TestPulseClient::producePlots(ProcessType)
  {
    using namespace std;

    MESetMulti& meQuality(static_cast<MESetMulti&>(MEs_.at("Quality")));
    MESetMulti& meAmplitudeRMS(static_cast<MESetMulti&>(MEs_.at("AmplitudeRMS")));
    MESetMulti& meQualitySummary(static_cast<MESetMulti&>(MEs_.at("QualitySummary")));
    MESetMulti& mePNQualitySummary(static_cast<MESetMulti&>(MEs_.at("PNQualitySummary")));

    MESetMulti const& sAmplitude(static_cast<MESetMulti const&>(sources_.at("Amplitude")));
    MESetMulti const& sPNAmplitude(static_cast<MESetMulti const&>(sources_.at("PNAmplitude")));

    for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      meQuality.use(gainItr->second);
      meQualitySummary.use(gainItr->second);
      meAmplitudeRMS.use(gainItr->second);

      sAmplitude.use(gainItr->second);

      uint32_t mask(0);
      switch(gainItr->first){
      case 1:
        mask |= (1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR);
        break;
      case 6:
        mask |= (1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR);
        break;
      case 12:
        mask |= (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR);
        break;
      default:
        break;
      }

      MESet::iterator qEnd(meQuality.end());
      MESet::iterator rItr(meAmplitudeRMS);
      MESet::const_iterator aItr(sAmplitude);
      for(MESet::iterator qItr(meQuality.beginChannel()); qItr != qEnd; qItr.toNextChannel()){

        DetId id(qItr->getId());

        bool doMask(meQuality.maskMatches(id, mask, statusManager_));

        aItr = qItr;
        rItr = qItr;

        float entries(aItr->getBinEntries());

        if(entries < minChannelEntries_){
          qItr->setBinContent(doMask ? kMUnknown : kUnknown);
          continue;
        }

        float amp(aItr->getBinContent());
        float rms(aItr->getBinError() * sqrt(entries));

        rItr->setBinContent(rms);

        if(amp < amplitudeThreshold_[gainItr->second] || rms > toleranceRMS_[gainItr->second])
          qItr->setBinContent(doMask ? kMBad : kBad);
        else
          qItr->setBinContent(doMask ? kMGood : kGood);
      }

      towerAverage_(meQualitySummary, meQuality, 0.2);
    }

    for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      mePNQualitySummary.use(gainItr->second);

      sPNAmplitude.use(gainItr->second);

      uint32_t mask(0);
      switch(gainItr->first){
      case 1:
        mask |= (1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR);
        break;
      case 16:
        mask |= (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR);
        break;
      default:
        break;
      }

      for(unsigned iDCC(0); iDCC < nDCC; ++iDCC){

        if(memDCCIndex(iDCC + 1) == unsigned(-1)) continue;

        for(unsigned iPN(0); iPN < 10; ++iPN){
          int subdet(0);
          if(iDCC >= kEBmLow && iDCC <= kEBpHigh) subdet = EcalBarrel;
          else subdet = EcalEndcap;

          EcalPnDiodeDetId id(subdet, iDCC + 1, iPN + 1);

          bool doMask(mePNQualitySummary.maskMatches(id, mask, statusManager_));

          float amp(sPNAmplitude.getBinContent(id));
          float entries(sPNAmplitude.getBinEntries(id));
          float rms(sPNAmplitude.getBinError(id) * sqrt(entries));

          if(entries < minChannelEntries_){
            mePNQualitySummary.setBinContent(id, doMask ? kMUnknown : kUnknown);
            continue;
          }

          if(amp < PNAmplitudeThreshold_[gainItr->second] || rms > tolerancePNRMS_[gainItr->second])
            mePNQualitySummary.setBinContent(id, doMask ? kMBad : kBad);
          else
            mePNQualitySummary.setBinContent(id, doMask ? kMGood : kGood);
        }
      }
    }
  }

  DEFINE_ECALDQM_WORKER(TestPulseClient);
}
