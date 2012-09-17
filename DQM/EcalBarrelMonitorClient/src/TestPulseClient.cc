#include "../interface/TestPulseClient.h"

#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include <iomanip>

namespace ecaldqm
{

  TestPulseClient::TestPulseClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "TestPulseClient"),
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

    stringstream ss;

    amplitudeThreshold_.resize(iMEGain);
    toleranceRMS_.resize(iMEGain);

    for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      ss.str("");
      ss << "G" << setfill('0') << setw(2) << gainItr->first;

      amplitudeThreshold_[gainItr->second] = _workerParams.getUntrackedParameter<double>("amplitudeThreshold" + ss.str());
      toleranceRMS_[gainItr->second] = _workerParams.getUntrackedParameter<double>("toleranceRMS" + ss.str());
    }

    PNAmplitudeThreshold_.resize(iMEPNGain);
    tolerancePNRMS_.resize(iMEPNGain);

    for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      ss.str("");
      ss << "G" << setfill('0') << setw(2) << gainItr->first;

      PNAmplitudeThreshold_[gainItr->second] = _workerParams.getUntrackedParameter<double>("PNAmplitudeThreshold" + ss.str());
      tolerancePNRMS_[gainItr->second] = _workerParams.getUntrackedParameter<double>("tolerancePNRMS" + ss.str());
    }

    map<string, string> replacements;

    unsigned apdPlots[] = {kQuality, kAmplitudeRMS, kQualitySummary};
    for(unsigned iS(0); iS < sizeof(apdPlots) / sizeof(unsigned); ++iS){
      unsigned plot(apdPlots[iS]);
      MESetMulti* multi(static_cast<MESetMulti*>(MEs_[plot]));

      for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << setfill('0') << setw(2) << gainItr->first;
        replacements["gain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    unsigned pnPlots[] = {kPNQualitySummary};
    for(unsigned iS(0); iS < sizeof(pnPlots) / sizeof(unsigned); ++iS){
      unsigned plot(pnPlots[iS]);
      MESetMulti* multi(static_cast<MESetMulti*>(MEs_[plot]));

      for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << setfill('0') << setw(2) << gainItr->first;
        replacements["pngain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    unsigned apdSources[] = {kAmplitude};
    for(unsigned iS(0); iS < sizeof(apdSources) / sizeof(unsigned); ++iS){
      unsigned plot(apdSources[iS]);
      MESetMulti const* multi(static_cast<MESetMulti const*>(sources_[plot]));

      for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << setfill('0') << setw(2) << gainItr->first;
        replacements["gain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    unsigned pnSources[] = {kPNAmplitude};
    for(unsigned iS(0); iS < sizeof(pnSources) / sizeof(unsigned); ++iS){
      unsigned plot(pnSources[iS]);
      MESetMulti const* multi(static_cast<MESetMulti const*>(sources_[plot]));

      for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << setfill('0') << setw(2) << gainItr->first;
        replacements["pngain"] = ss.str();

        multi->formPath(replacements);
      }
    }
  }

  void
  TestPulseClient::beginRun(edm::Run const&, edm::EventSetup const&)
  {
    for(unsigned iME(0); iME < gainToME_.size(); ++iME){
      static_cast<MESetMulti*>(MEs_[kQuality])->use(iME);
      static_cast<MESetMulti*>(MEs_[kQualitySummary])->use(iME);

      MEs_[kAmplitudeRMS]->resetAll(-1.);
      MEs_[kQuality]->resetAll(-1.);
      MEs_[kQualitySummary]->resetAll(-1.);

      MEs_[kQuality]->reset(kUnknown);
      MEs_[kQualitySummary]->reset(kUnknown);
    }

    for(unsigned iME(0); iME < pnGainToME_.size(); ++iME){
      static_cast<MESetMulti*>(MEs_[kPNQualitySummary])->use(iME);

      MEs_[kPNQualitySummary]->resetAll(-1.);

      MEs_[kPNQualitySummary]->reset(kUnknown);
    }
  }

  void
  TestPulseClient::producePlots()
  {
    using namespace std;

    for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      static_cast<MESetMulti*>(MEs_[kQuality])->use(gainItr->second);
      static_cast<MESetMulti*>(MEs_[kQualitySummary])->use(gainItr->second);
      static_cast<MESetMulti*>(MEs_[kAmplitudeRMS])->use(gainItr->second);

      static_cast<MESetMulti const*>(sources_[kAmplitude])->use(gainItr->second);

      MEs_[kAmplitudeRMS]->reset();

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

      MESet::iterator qEnd(MEs_[kQuality]->end());
      MESet::const_iterator aItr(sources_[kAmplitude]);
      for(MESet::iterator qItr(MEs_[kQuality]->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

        DetId id(qItr->getId());

        bool doMask(applyMask_(kQuality, id, mask));

        aItr = qItr;

        float entries(aItr->getBinEntries());

        if(entries < 1.){
          qItr->setBinContent(doMask ? kMUnknown : kUnknown);
          continue;
        }

        float amp(aItr->getBinContent());
        float rms(aItr->getBinError() * sqrt(entries));

        MEs_[kAmplitudeRMS]->setBinContent(id, rms);

        if(amp < amplitudeThreshold_[gainItr->second] || rms > toleranceRMS_[gainItr->second])
          qItr->setBinContent(doMask ? kMBad : kBad);
        else
          qItr->setBinContent(doMask ? kMGood : kGood);
      }

      towerAverage_(kQualitySummary, kQuality, 0.2);
    }

    for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      static_cast<MESetMulti*>(MEs_[kPNQualitySummary])->use(gainItr->second);

      static_cast<MESetMulti const*>(sources_[kPNAmplitude])->use(gainItr->second);

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

          bool doMask(applyMask_(kPNQualitySummary, id, mask));

          float amp(sources_[kPNAmplitude]->getBinContent(id));
          float entries(sources_[kPNAmplitude]->getBinEntries(id));
          float rms(sources_[kPNAmplitude]->getBinError(id) * sqrt(entries));

          if(entries < 1.){
            MEs_[kPNQualitySummary]->setBinContent(id, doMask ? kMUnknown : kUnknown);
            continue;
          }

          if(amp < PNAmplitudeThreshold_[gainItr->second] || rms > tolerancePNRMS_[gainItr->second])
            MEs_[kPNQualitySummary]->setBinContent(id, doMask ? kMBad : kBad);
          else
            MEs_[kPNQualitySummary]->setBinContent(id, doMask ? kMGood : kGood);
        }
      }
    }
  }

  /*static*/
  void
  TestPulseClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Quality"] = kQuality;
    _nameToIndex["AmplitudeRMS"] = kAmplitudeRMS;
    _nameToIndex["QualitySummary"] = kQualitySummary;
    _nameToIndex["PNQualitySummary"] = kPNQualitySummary;

    _nameToIndex["Amplitude"] = kAmplitude;
    _nameToIndex["PNAmplitude"] = kPNAmplitude;
  }

  DEFINE_ECALDQM_WORKER(TestPulseClient);
}
