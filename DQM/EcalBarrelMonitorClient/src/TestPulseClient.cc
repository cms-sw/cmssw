#include "../interface/TestPulseClient.h"
#include "../interface/EcalDQMClientUtils.h"

#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include <iomanip>

namespace ecaldqm
{

  TestPulseClient::TestPulseClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "TestPulseClient"),
    minChannelEntries_(_workerParams.getUntrackedParameter<int>("minChannelEntries")),
    amplitudeThreshold_(0),
    toleranceRMS_(0),
    PNAmplitudeThreshold_(0),
    tolerancePNRMS_(0)
  {
    using namespace std;

    vector<int> MGPAGains(_commonParams.getUntrackedParameter<vector<int> >("MGPAGains"));
    vector<int> MGPAGainsPN(_commonParams.getUntrackedParameter<vector<int> >("MGPAGainsPN"));

    unsigned iMEGain(0);
    for(vector<int>::iterator gainItr(MGPAGains.begin()); gainItr != MGPAGains.end(); ++gainItr){
      if(*gainItr != 1 && *gainItr != 6 && *gainItr != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain" << endl;
      gainToME_[*gainItr] = iMEGain++;
    }

    unsigned iMEPNGain(0);
    for(vector<int>::iterator gainItr(MGPAGainsPN.begin()); gainItr != MGPAGainsPN.end(); ++gainItr){
      if(*gainItr != 1 && *gainItr != 16) throw cms::Exception("InvalidConfiguration") << "PN diode gain" << endl;	
      pnGainToME_[*gainItr] = iMEPNGain++;
    }

    amplitudeThreshold_.resize(iMEGain);
    toleranceRMS_.resize(iMEGain);

    std::vector<double> inAmplitudeThreshold = _workerParams.getUntrackedParameter<std::vector<double> >("amplitudeThreshold");
    std::vector<double> inToleranceRMS = _workerParams.getUntrackedParameter<std::vector<double> >("toleranceRMS");

    for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
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

    PNAmplitudeThreshold_.resize(iMEPNGain);
    tolerancePNRMS_.resize(iMEPNGain);

    std::vector<double> inPNAmplitudeThreshold = _workerParams.getUntrackedParameter<std::vector<double> >("PNAmplitudeThreshold");
    std::vector<double> inTolerancePNRMS = _workerParams.getUntrackedParameter<std::vector<double> >("tolerancePNRMS");

    for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
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

    map<string, string> replacements;
    stringstream ss;

    std::string apdPlots[] = {"Quality", "AmplitudeRMS", "QualitySummary"};
    for(unsigned iS(0); iS < sizeof(apdPlots) / sizeof(std::string); ++iS){
      std::string& plot(apdPlots[iS]);
      MESetMulti* multi(static_cast<MESetMulti*>(MEs_[plot]));

      for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << setfill('0') << setw(2) << gainItr->first;
        replacements["gain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    std::string pnPlots[] = {"PNQualitySummary"};
    for(unsigned iS(0); iS < sizeof(pnPlots) / sizeof(std::string); ++iS){
      std::string& plot(pnPlots[iS]);
      MESetMulti* multi(static_cast<MESetMulti*>(MEs_[plot]));

      for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << setfill('0') << setw(2) << gainItr->first;
        replacements["pngain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    std::string apdSources[] = {"Amplitude"};
    for(unsigned iS(0); iS < sizeof(apdSources) / sizeof(std::string); ++iS){
      std::string& plot(apdSources[iS]);
      MESetMulti const* multi(static_cast<MESetMulti const*>(sources_[plot]));

      for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << setfill('0') << setw(2) << gainItr->first;
        replacements["gain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    std::string pnSources[] = {"PNAmplitude"};
    for(unsigned iS(0); iS < sizeof(pnSources) / sizeof(std::string); ++iS){
      std::string& plot(pnSources[iS]);
      MESetMulti const* multi(static_cast<MESetMulti const*>(sources_[plot]));

      for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << setfill('0') << setw(2) << gainItr->first;
        replacements["pngain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
    qualitySummaries_.insert("PNQualitySummary");
  }

  void
  TestPulseClient::producePlots()
  {
    using namespace std;

    MESetMulti* meQuality(static_cast<MESetMulti*>(MEs_["Quality"]));
    MESetMulti* meAmplitudeRMS(static_cast<MESetMulti*>(MEs_["AmplitudeRMS"]));
    MESetMulti* meQualitySummary(static_cast<MESetMulti*>(MEs_["QualitySummary"]));
    MESetMulti* mePNQualitySummary(static_cast<MESetMulti*>(MEs_["PNQualitySummary"]));

    MESetMulti const* sAmplitude(static_cast<MESetMulti const*>(sources_["Amplitude"]));
    MESetMulti const* sPNAmplitude(static_cast<MESetMulti const*>(sources_["PNAmplitude"]));

    for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      meQuality->use(gainItr->second);
      meQualitySummary->use(gainItr->second);
      meAmplitudeRMS->use(gainItr->second);

      sAmplitude->use(gainItr->second);

      meAmplitudeRMS->reset();

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

      MESet::iterator qEnd(meQuality->end());
      MESet::const_iterator aItr(sAmplitude);
      for(MESet::iterator qItr(meQuality->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

        DetId id(qItr->getId());

        bool doMask(applyMask(meQuality->getBinType(), id, mask));

        aItr = qItr;

        float entries(aItr->getBinEntries());

        if(entries < minChannelEntries_){
          qItr->setBinContent(doMask ? kMUnknown : kUnknown);
          continue;
        }

        float amp(aItr->getBinContent());
        float rms(aItr->getBinError() * sqrt(entries));

        meAmplitudeRMS->setBinContent(id, rms);

        if(amp < amplitudeThreshold_[gainItr->second] || rms > toleranceRMS_[gainItr->second])
          qItr->setBinContent(doMask ? kMBad : kBad);
        else
          qItr->setBinContent(doMask ? kMGood : kGood);
      }

      towerAverage_(meQualitySummary, meQuality, 0.2);
    }

    for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      mePNQualitySummary->use(gainItr->second);

      sPNAmplitude->use(gainItr->second);

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

      for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){

        if(memDCCIndex(iDCC + 1) == unsigned(-1)) continue;

        for(unsigned iPN(0); iPN < 10; ++iPN){
          int subdet(0);
          if(iDCC >= kEBmLow && iDCC <= kEBpHigh) subdet = EcalBarrel;
          else subdet = EcalEndcap;

          EcalPnDiodeDetId id(subdet, iDCC + 1, iPN + 1);

          bool doMask(applyMask(mePNQualitySummary->getBinType(), id, mask));

          float amp(sPNAmplitude->getBinContent(id));
          float entries(sPNAmplitude->getBinEntries(id));
          float rms(sPNAmplitude->getBinError(id) * sqrt(entries));

          if(entries < minChannelEntries_){
            mePNQualitySummary->setBinContent(id, doMask ? kMUnknown : kUnknown);
            continue;
          }

          if(amp < PNAmplitudeThreshold_[gainItr->second] || rms > tolerancePNRMS_[gainItr->second])
            mePNQualitySummary->setBinContent(id, doMask ? kMBad : kBad);
          else
            mePNQualitySummary->setBinContent(id, doMask ? kMGood : kGood);
        }
      }
    }
  }

  DEFINE_ECALDQM_WORKER(TestPulseClient);
}
