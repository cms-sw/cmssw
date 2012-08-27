#include "../interface/LedClient.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

#include <cmath>

namespace ecaldqm {

  LedClient::LedClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "LedClient"),
    wlToME_(),
    wlGainToME_(),
    minChannelEntries_(_workerParams.getUntrackedParameter<int>("minChanelEntries")),
    expectedAmplitude_(0),
    amplitudeThreshold_(0),
    amplitudeRMSThreshold_(0),
    expectedTiming_(0),
    timingThreshold_(0),
    timingRMSThreshold_(0),
    expectedPNAmplitude_(0),
    pnAmplitudeThreshold_(0),
    pnAmplitudeRMSThreshold_(0),
    towerThreshold_(_workerParams.getUntrackedParameter<double>("towerThreshold"))
  {
    using namespace std;

    vector<int> MGPAGainsPN(_commonParams.getUntrackedParameter<vector<int> >("MGPAGainsPN"));
    vector<int> ledWavelengths(_commonParams.getUntrackedParameter<vector<int> >("ledWavelengths"));

    unsigned iMEWL(0);
    unsigned iMEWLG(0);
    for(vector<int>::iterator wlItr(ledWavelengths.begin()); wlItr != ledWavelengths.end(); ++wlItr){
      if(*wlItr <= 0 || *wlItr >= 5) throw cms::Exception("InvalidConfiguration") << "Led Wavelength" << endl;
      wlToME_[*wlItr] = iMEWL++;

      for(vector<int>::iterator gainItr(MGPAGainsPN.begin()); gainItr != MGPAGainsPN.end(); ++gainItr){
        if(*gainItr != 1 && *gainItr != 16) throw cms::Exception("InvalidConfiguration") << "PN diode gain" << endl;
        wlGainToME_[make_pair(*wlItr, *gainItr)] = iMEWLG++;
      }
    }

    stringstream ss;

    expectedAmplitude_.resize(iMEWL);
    amplitudeThreshold_.resize(iMEWL);
    amplitudeRMSThreshold_.resize(iMEWL);
    expectedTiming_.resize(iMEWL);
    timingThreshold_.resize(iMEWL);
    timingRMSThreshold_.resize(iMEWL);

    for(map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
      ss.str("");
      ss << "L" << wlItr->first;

      expectedAmplitude_[wlItr->second] = _workerParams.getUntrackedParameter<double>("expectedAmplitude" + ss.str());
      amplitudeThreshold_[wlItr->second] = _workerParams.getUntrackedParameter<double>("amplitudeThreshold" + ss.str());
      amplitudeRMSThreshold_[wlItr->second] = _workerParams.getUntrackedParameter<double>("amplitudeRMSThreshold" + ss.str());
      expectedTiming_[wlItr->second] = _workerParams.getUntrackedParameter<double>("expectedTiming" + ss.str());
      timingThreshold_[wlItr->second] = _workerParams.getUntrackedParameter<double>("timingThreshold" + ss.str());
      timingRMSThreshold_[wlItr->second] = _workerParams.getUntrackedParameter<double>("timingRMSThreshold" + ss.str());
    }

    expectedPNAmplitude_.resize(iMEWLG);
    pnAmplitudeThreshold_.resize(iMEWLG);
    pnAmplitudeRMSThreshold_.resize(iMEWLG);

    for(map<pair<int, int>, unsigned>::iterator wlGainItr(wlGainToME_.begin()); wlGainItr != wlGainToME_.end(); ++wlGainItr){
      ss.str("");
      ss << "L" << wlGainItr->first.first << "G" << wlGainItr->first.second;

      expectedPNAmplitude_[wlGainItr->second] = _workerParams.getUntrackedParameter<double>("expectedPNAmplitude" + ss.str());
      pnAmplitudeThreshold_[wlGainItr->second] = _workerParams.getUntrackedParameter<double>("pnAmplitudeThreshold" + ss.str());
      pnAmplitudeRMSThreshold_[wlGainItr->second] = _workerParams.getUntrackedParameter<double>("pnAmplitudeRMSThreshold" + ss.str());
    }

    map<string, string> replacements;

    unsigned apdPlots[] = {kQuality, kAmplitudeMean, kAmplitudeRMS, kTimingMean, kTimingRMS, kQualitySummary, kPNQualitySummary};
    for(unsigned iS(0); iS < sizeof(apdPlots) / sizeof(unsigned); ++iS){
      unsigned plot(apdPlots[iS]);
      MESet* temp(MEs_[plot]);
      MESetMulti* meSet(new MESetMulti(*temp, iMEWL));

      for(map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
        meSet->use(wlItr->second);

        ss.str("");
        ss << wlItr->first;
        replacements["wl"] = ss.str();

        meSet->formPath(replacements);
      }

      MEs_[plot] = meSet;
      delete temp;
    }

    unsigned pnPlots[] = {kPNAmplitudeMean, kPNAmplitudeRMS};
    for(unsigned iS(0); iS < sizeof(pnPlots) / sizeof(unsigned); ++iS){
      unsigned plot(pnPlots[iS]);
      MESet* temp(MEs_[plot]);
      MESetMulti* meSet(new MESetMulti(*temp, iMEWLG));

      for(map<pair<int, int>, unsigned>::iterator wlGainItr(wlGainToME_.begin()); wlGainItr != wlGainToME_.end(); ++wlGainItr){
        meSet->use(wlGainItr->second);

        ss.str("");
        ss << wlGainItr->first.first;
        replacements["wl"] = ss.str();

	ss.str("");
	ss << wlGainItr->first.second;
	replacements["pngain"] = ss.str();

        meSet->formPath(replacements);
      }

      MEs_[plot] = meSet;
      delete temp;
    }

    unsigned apdSources[] = {kAmplitude, kTiming};
    for(unsigned iS(0); iS < sizeof(apdSources) / sizeof(unsigned); ++iS){
      unsigned plot(apdSources[iS]);
      MESet const* temp(sources_[plot]);
      MESetMulti const* meSet(new MESetMulti(*temp, iMEWL));

      for(map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
        meSet->use(wlItr->second);

        ss.str("");
        ss << wlItr->first;
        replacements["wl"] = ss.str();

        meSet->formPath(replacements);
      }

      sources_[plot] = meSet;
      delete temp;
    }

    unsigned pnSources[] = {kPNAmplitude};
    for(unsigned iS(0); iS < sizeof(pnSources) / sizeof(unsigned); ++iS){
      unsigned plot(pnSources[iS]);
      MESet const* temp(sources_[plot]);
      MESetMulti const* meSet(new MESetMulti(*temp, iMEWLG));

      for(map<pair<int, int>, unsigned>::iterator wlGainItr(wlGainToME_.begin()); wlGainItr != wlGainToME_.end(); ++wlGainItr){
        meSet->use(wlGainItr->second);

        ss.str("");
        ss << wlGainItr->first.first;
        replacements["wl"] = ss.str();

	ss.str("");
	ss << wlGainItr->first.second;
	replacements["pngain"] = ss.str();

        meSet->formPath(replacements);
      }

      sources_[plot] = meSet;
      delete temp;
    }
  }

  void
  LedClient::beginRun(const edm::Run &, const edm::EventSetup &)
  {
    for(unsigned iME(0); iME < wlToME_.size(); ++iME){
      static_cast<MESetMulti*>(MEs_[kQuality])->use(iME);
      static_cast<MESetMulti*>(MEs_[kQualitySummary])->use(iME);
      static_cast<MESetMulti*>(MEs_[kPNQualitySummary])->use(iME);

      MEs_[kQuality]->resetAll(-1.);
      MEs_[kQualitySummary]->resetAll(-1.);
      MEs_[kPNQualitySummary]->resetAll(-1.);
    }
  }

  void
  LedClient::producePlots()
  {
    uint32_t mask(1 << EcalDQMStatusHelper::LED_MEAN_ERROR |
                  1 << EcalDQMStatusHelper::LED_RMS_ERROR |
                  1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR |
                  1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR);

    for(std::map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
      static_cast<MESetMulti*>(MEs_[kQuality])->use(wlItr->second);
      static_cast<MESetMulti*>(MEs_[kQualitySummary])->use(wlItr->second);
      static_cast<MESetMulti*>(MEs_[kAmplitudeMean])->use(wlItr->second);
      static_cast<MESetMulti*>(MEs_[kAmplitudeRMS])->use(wlItr->second);
      static_cast<MESetMulti*>(MEs_[kTimingMean])->use(wlItr->second);
      static_cast<MESetMulti*>(MEs_[kTimingRMS])->use(wlItr->second);

      static_cast<MESetMulti const*>(sources_[kAmplitude])->use(wlItr->second);
      static_cast<MESetMulti const*>(sources_[kTiming])->use(wlItr->second);

      MEs_[kAmplitudeMean]->reset();
      MEs_[kAmplitudeRMS]->reset();
      MEs_[kTimingMean]->reset();
      MEs_[kTimingRMS]->reset();

      MESet::iterator qEnd(MEs_[kQuality]->end());

      MESet::const_iterator tItr(sources_[kTiming]);
      MESet::const_iterator aItr(sources_[kAmplitude]);
      for(MESet::iterator qItr(MEs_[kQuality]->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

        DetId id(qItr->getId());

        aItr = qItr;

        float aEntries(aItr->getBinEntries());

        if(aEntries < minChannelEntries_){
          qItr->setBinContent(maskQuality_(qItr, mask, 2));
          continue;
        }

        float aMean(aItr->getBinContent());
        float aRms(aItr->getBinError() * sqrt(aEntries));

        MEs_[kAmplitudeMean]->fill(id, aMean);
        MEs_[kAmplitudeRMS]->fill(id, aRms);

        tItr = qItr;

        float tEntries(tItr->getBinEntries());

        if(tEntries < minChannelEntries_) continue;

        float tMean(tItr->getBinContent());
        float tRms(tItr->getBinError() * sqrt(tEntries));

        MEs_[kTimingMean]->fill(id, tMean);
        MEs_[kTimingRMS]->fill(id, tRms);

        if(abs(aMean - expectedAmplitude_[wlItr->second]) > amplitudeThreshold_[wlItr->second] || aRms > amplitudeRMSThreshold_[wlItr->second] ||
           abs(tMean - expectedTiming_[wlItr->second]) > timingThreshold_[wlItr->second] || tRms > timingRMSThreshold_[wlItr->second])
          qItr->setBinContent(maskQuality_(qItr, mask, 0));
        else
          qItr->setBinContent(maskQuality_(qItr, mask, 1));

      }

      towerAverage_(kQualitySummary, kQuality, 0.5);
    }

    for(std::map<std::pair<int, int>, unsigned>::iterator wlGainItr(wlGainToME_.begin()); wlGainItr != wlGainToME_.end(); ++wlGainItr){
      static_cast<MESetMulti*>(MEs_[kPNAmplitudeMean])->use(wlGainItr->second);
      static_cast<MESetMulti*>(MEs_[kPNAmplitudeRMS])->use(wlGainItr->second);
      static_cast<MESetMulti*>(MEs_[kPNQualitySummary])->use(wlToME_[wlGainItr->first.first]);

      static_cast<MESetMulti const*>(sources_[kPNAmplitude])->use(wlGainItr->second);

      MEs_[kPNAmplitudeMean]->reset();
      MEs_[kPNAmplitudeRMS]->reset();

      MESet::const_iterator aEnd(sources_[kPNAmplitude]->end());
      for(MESet::const_iterator aItr(sources_[kPNAmplitude]->beginChannel()); aItr != aEnd; aItr.toNextChannel()){

        EcalPnDiodeDetId id(aItr->getId());

        float pEntries(aItr->getBinEntries());

        if(pEntries < minChannelEntries_){
          MEs_[kPNQualitySummary]->setBinContent(id, maskQuality_(kPNQualitySummary, id, mask, 2));
          continue;
        }

        float pMean(aItr->getBinContent());
        float pRms(aItr->getBinError() * sqrt(pEntries));

        MEs_[kPNAmplitudeMean]->fill(id, pMean);
        MEs_[kPNAmplitudeRMS]->fill(id, pRms);

        if(abs(pMean - expectedPNAmplitude_[wlGainItr->second]) > pnAmplitudeThreshold_[wlGainItr->second] || pRms > pnAmplitudeRMSThreshold_[wlGainItr->second])
          MEs_[kPNQualitySummary]->setBinContent(id, maskQuality_(kPNQualitySummary, id, mask, 0));
        else
          MEs_[kPNQualitySummary]->setBinContent(id, maskQuality_(kPNQualitySummary, id, mask, 1));
      }
    }
  }

  /*static*/
  void
  LedClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Quality"] = kQuality;
    _nameToIndex["AmplitudeMean"] = kAmplitudeMean;
    _nameToIndex["AmplitudeRMS"] = kAmplitudeRMS;
    _nameToIndex["TimingMean"] = kTimingMean;
    _nameToIndex["TimingRMS"] = kTimingRMS;
    _nameToIndex["PNAmplitudeMean"] = kPNAmplitudeMean;
    _nameToIndex["PNAmplitudeRMS"] = kPNAmplitudeRMS;
    _nameToIndex["QualitySummary"] = kQualitySummary;
    _nameToIndex["PNQualitySummary"] = kPNQualitySummary;

    _nameToIndex["Amplitude"] = kAmplitude;
    _nameToIndex["Timing"] = kTiming;
    _nameToIndex["PNAmplitude"] = kPNAmplitude;
  }

  DEFINE_ECALDQM_WORKER(LedClient);
}
