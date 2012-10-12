#include "../interface/LedTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

namespace ecaldqm {

  LedTask::LedTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "LedTask"),
    wlToME_(),
    pnAmp_(),
    emptyLS_(0),
    emptyLSLimit_(_workerParams.getUntrackedParameter<int>("emptyLSLimit"))
  {
    using namespace std;

    collectionMask_ = 
      (0x1 << kEcalRawData) |
      (0x1 << kEEDigi) |
      (0x1 << kPnDiodeDigi) |
      (0x1 << kEELaserLedUncalibRecHit);

    for(unsigned iD(0); iD < BinService::nEEDCC; ++iD){
      enable_[iD] = false;
      wavelength_[iD] = 0;
      rtHalf_[iD] = 0;
    }

    vector<int> ledWavelengths(_commonParams.getUntrackedParameter<vector<int> >("ledWavelengths"));

    unsigned iMEWL(0);
    for(vector<int>::iterator wlItr(ledWavelengths.begin()); wlItr != ledWavelengths.end(); ++wlItr){
      if(*wlItr <= 0 || *wlItr >= 3) throw cms::Exception("InvalidConfiguration") << "Led Wavelength" << endl;
      wlToME_[*wlItr] = iMEWL++;
    }

    map<string, string> replacements;
    stringstream ss;

    unsigned wlPlots[] = {kAmplitude, kAmplitudeSummary, kTiming, kShape, kAOverP, kPNAmplitude};
    for(unsigned iS(0); iS < sizeof(wlPlots) / sizeof(unsigned); ++iS){
      unsigned plot(wlPlots[iS]);
      MESetMulti* multi(static_cast<MESetMulti*>(MEs_[plot]));

      for(map<int, unsigned>::iterator wlItr(wlToME_.begin()); wlItr != wlToME_.end(); ++wlItr){
        multi->use(wlItr->second);

        ss.str("");
        ss << wlItr->first;
        replacements["wl"] = ss.str();

        multi->formPath(replacements);
      }
    }
  }

  void
  LedTask::setDependencies(DependencySet& _dependencies)
  {
    _dependencies.push_back(Dependency(kEEDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kPnDiodeDigi, kEEDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kEELaserLedUncalibRecHit, kPnDiodeDigi, kEEDigi, kEcalRawData));
  }

  bool
  LedTask::filterRunType(const std::vector<short>& _runType)
  {
    bool enable(false);

    for(unsigned iDCC(0); iDCC < BinService::nDCC; iDCC++){
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - BinService::nEBDCC);
      if(_runType[iDCC] == EcalDCCHeaderBlock::LED_STD ||
	 _runType[iDCC] == EcalDCCHeaderBlock::LED_GAP){
	enable = true;
	enable_[index] = true;
      }
      else
        enable_[index] = false;
    }

    return enable;
  }

  void
  LedTask::beginRun(const edm::Run &, const edm::EventSetup &)
  {
    emptyLS_ = 0;
  }

  void
  LedTask::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
  {
    if(++emptyLS_ > emptyLSLimit_) emptyLS_ = -1;
  }

  void
  LedTask::beginEvent(const edm::Event &, const edm::EventSetup &)
  {
    pnAmp_.clear();
  }

  void
  LedTask::runOnRawData(EcalRawDataCollection const& _rawData)
  {
    for(EcalRawDataCollection::const_iterator rItr(_rawData.begin()); rItr != _rawData.end(); ++rItr){
      unsigned iDCC(rItr->id() - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - BinService::nEBDCC);

      if(!enable_[index]){
        wavelength_[index] = -1;
        rtHalf_[index] = -1;
        continue;
      }
      if(rItr->getEventSettings().wavelength == 0)
        wavelength_[index] = 1;
      else if(rItr->getEventSettings().wavelength == 2)
        wavelength_[index] = 2;
      else
        wavelength_[index] = -1;

      if(wlToME_.find(wavelength_[index]) == wlToME_.end())
        enable_[index] = false;

      rtHalf_[index] = rItr->getRtHalf();
    }
  }

  void
  LedTask::runOnDigis(const EcalDigiCollection &_digis)
  {
    int nReadouts[BinService::nEEDCC];
    int maxpos[BinService::nEEDCC][10];
    for(unsigned index(0); index < BinService::nEEDCC; ++index){
      nReadouts[index] = 0;
      for(int i(0); i < 10; i++) maxpos[index][i] = 0;
    }

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const DetId& id(digiItr->id());

      MEs_[kOccupancy]->fill(id);

      unsigned iDCC(dccId(id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - BinService::nEBDCC);

      if(!enable_[index]) continue;
      if(rtHalf(id) != rtHalf_[index]) continue;

      ++nReadouts[index];

      EcalDataFrame dataFrame(*digiItr);

      int iMax(-1);
      int max(0);
      int min(4096);
      for (int i(0); i < 10; i++) {
        int adc(dataFrame.sample(i).adc());
        if(adc > max){
          max = adc;
          iMax = i;
        }
        if(adc < min) min = adc;
      }
      if(iMax >= 0 && max - min > 3) // normal RMS of pedestal is ~2.5
        maxpos[index][iMax] += 1;
    }

    // signal existence check
    bool enable(false);
    bool ledOnExpected(emptyLS_ >= 0);

    for(unsigned index(0); index < BinService::nEEDCC; ++index){
      if(nReadouts[index] == 0){
        enable_[index] = false;
        continue;
      }

      int threshold(nReadouts[index] / 3);
      if(ledOnExpected) enable_[index] = false;

      for(int i(0); i < 10; i++){
        if(maxpos[index][i] > threshold){
          enable = true;
          enable_[index] = true;
          break;
        }
      }
    }

    if(enable) emptyLS_ = 0;
    else if(ledOnExpected) return;

    unsigned iME(-1);
    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const DetId& id(digiItr->id());

      unsigned iDCC(dccId(id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - BinService::nEBDCC);

      if(!enable_[index]) continue;
      if(rtHalf(id) != rtHalf_[index]) continue;

      if(iME != wlToME_[wavelength_[index]]){
        iME = wlToME_[wavelength_[index]];
        static_cast<MESetMulti*>(MEs_[kShape])->use(iME);
      }

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      for(int iSample(0); iSample < 10; iSample++)
	MEs_[kShape]->fill(id, iSample + 0.5, float(dataFrame.sample(iSample).adc()));

      EcalPnDiodeDetId pnidA(pnForCrystal(id, 'a'));
      EcalPnDiodeDetId pnidB(pnForCrystal(id, 'b'));
      if(pnidA.null() || pnidB.null()) continue;
      pnAmp_.insert(std::make_pair(pnidA.rawId(), 0.));
      pnAmp_.insert(std::make_pair(pnidB.rawId(), 0.));
    }
  }

  void
  LedTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
  {
    unsigned iME(-1);

    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      if(digiItr->sample(0).gainId() != 0 && digiItr->sample(0).gainId() != 1) continue;

      const EcalPnDiodeDetId& id(digiItr->id());

      std::map<uint32_t, float>::iterator ampItr(pnAmp_.find(id.rawId()));
      if(ampItr == pnAmp_.end()) continue;

      unsigned iDCC(dccId(id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - BinService::nEBDCC);

      float pedestal(0.);
      for(int iSample(0); iSample < 4; iSample++)
	pedestal += digiItr->sample(iSample).adc();
      pedestal /= 4.;

      float max(0.);
      for(int iSample(0); iSample < 50; iSample++){
	float amp(digiItr->sample(iSample).adc() - pedestal);
	if(amp > max) max = amp;
      }

      if(iME != wlToME_[wavelength_[index]]){
        iME = wlToME_[wavelength_[index]];
        static_cast<MESetMulti*>(MEs_[kPNAmplitude])->use(iME);
      }

      MEs_[kPNAmplitude]->fill(id, max);

      ampItr->second = max;
    }
  }

  void
  LedTask::runOnUncalibRecHits(const EcalUncalibratedRecHitCollection &_uhits)
  {
    using namespace std;

    unsigned iME(-1);

    for(EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr){
      EEDetId id(uhitItr->id());

      unsigned iDCC(dccId(id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - BinService::nEBDCC);

      if(!enable_[index]) continue;
      if(rtHalf(id) != rtHalf_[index]) continue;

      if(iME != wlToME_[wavelength_[index]]){
        iME = wlToME_[wavelength_[index]];
        static_cast<MESetMulti*>(MEs_[kAmplitude])->use(iME);
        static_cast<MESetMulti*>(MEs_[kAmplitudeSummary])->use(iME);
        static_cast<MESetMulti*>(MEs_[kTiming])->use(iME);
        static_cast<MESetMulti*>(MEs_[kAOverP])->use(iME);
      }

      float amp(max((double)uhitItr->amplitude(), 0.));
      float jitter(max((double)uhitItr->jitter() + 5.0, 0.));

      MEs_[kAmplitude]->fill(id, amp);
      MEs_[kAmplitudeSummary]->fill(id, amp);
      MEs_[kTiming]->fill(id, jitter);

      float aop(0.);

      map<uint32_t, float>::iterator ampItrA(pnAmp_.find(pnForCrystal(id, 'a')));
      map<uint32_t, float>::iterator ampItrB(pnAmp_.find(pnForCrystal(id, 'b')));
      if(ampItrA == pnAmp_.end() && ampItrB == pnAmp_.end()) continue;
      else if(ampItrB == pnAmp_.end()) aop = amp / ampItrA->second;
      else if(ampItrA == pnAmp_.end()) aop = amp / ampItrB->second;
      else aop = amp / (ampItrA->second + ampItrB->second) * 2.;

      MEs_[kAOverP]->fill(id, aop);
    }
  }

  /*static*/
  void
  LedTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Amplitude"] = kAmplitude;
    _nameToIndex["AmplitudeSummary"] = kAmplitudeSummary;
    _nameToIndex["Occupancy"] = kOccupancy;
    _nameToIndex["Shape"] = kShape;
    _nameToIndex["Timing"] = kTiming;
    _nameToIndex["AOverP"] = kAOverP;
    _nameToIndex["PNAmplitude"] = kPNAmplitude;
  }

  DEFINE_ECALDQM_WORKER(LedTask);
}

