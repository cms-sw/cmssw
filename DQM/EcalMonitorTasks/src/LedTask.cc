#include "../interface/LedTask.h"

#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm {

  LedTask::LedTask() :
    DQWorkerTask(),
    wlToME_(),
    pnAmp_(),
    emptyLS_(0),
    emptyLSLimit_(0)
  {
    std::fill_n(enable_, nEEDCC, false);
    std::fill_n(wavelength_, nEEDCC, 0);
    std::fill_n(rtHalf_, nEEDCC, 0);
  }

  void
  LedTask::setParams(edm::ParameterSet const& _params)
  {
    emptyLSLimit_ = _params.getUntrackedParameter<int>("emptyLSLimit");

    std::vector<int> ledWavelengths(_params.getUntrackedParameter<std::vector<int> >("ledWavelengths"));

    MESet::PathReplacements repl;

    MESetMulti& amplitude(static_cast<MESetMulti&>(MEs_.at("Amplitude")));
    unsigned nWL(ledWavelengths.size());
    for(unsigned iWL(0); iWL != nWL; ++iWL){
      int wl(ledWavelengths[iWL]);
      if(wl != 1 && wl != 2) throw cms::Exception("InvalidConfiguration") << "Led Wavelength";
      repl["wl"] = std::to_string(wl);
      wlToME_[wl] = amplitude.getIndex(repl);
    }
  }

  void
  LedTask::addDependencies(DependencySet& _dependencies)
  {
    _dependencies.push_back(Dependency(kEEDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kPnDiodeDigi, kEEDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kEELaserLedUncalibRecHit, kPnDiodeDigi, kEEDigi, kEcalRawData));
  }

  bool
  LedTask::filterRunType(short const* _runType)
  {
    bool enable(false);

    for(unsigned iDCC(0); iDCC != nDCC; iDCC++){
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - nEBDCC);
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
  LedTask::beginRun(edm::Run const&, edm::EventSetup const&)
  {
    emptyLS_ = 0;
  }

  void
  LedTask::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
    if(++emptyLS_ > emptyLSLimit_) emptyLS_ = -1;
  }

  void
  LedTask::beginEvent(edm::Event const&, edm::EventSetup const&)
  {
    pnAmp_.clear();
  }

  void
  LedTask::runOnRawData(EcalRawDataCollection const& _rawData)
  {
    for(EcalRawDataCollection::const_iterator rItr(_rawData.begin()); rItr != _rawData.end(); ++rItr){
      unsigned iDCC(rItr->id() - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - nEBDCC);

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
  LedTask::runOnDigis(EEDigiCollection const& _digis)
  {
    MESet& meOccupancy(MEs_.at("Occupancy"));
    MESet& meShape(MEs_.at("Shape"));
    MESet& meSignalRate(MEs_.at("SignalRate"));

    int nReadouts[nEEDCC];
    int maxpos[nEEDCC][10];
    for(unsigned index(0); index < nEEDCC; ++index){
      nReadouts[index] = 0;
      for(int i(0); i < 10; i++) maxpos[index][i] = 0;
    }

    for(EEDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const DetId& id(digiItr->id());

      unsigned iDCC(dccId(id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - nEBDCC);

      if(!enable_[index]) continue;
      if(rtHalf(id) != rtHalf_[index]) continue;

      meOccupancy.fill(id);

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

    unsigned iME(-1);

    for(int index(0); index < nEEDCC; ++index){
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

      if(iME != wlToME_[wavelength_[index]]){
        iME = wlToME_[wavelength_[index]];
        static_cast<MESetMulti&>(meSignalRate).use(iME);
      }

      meSignalRate.fill((index <= kEEmHigh ? index : index + nEBDCC) + 1, enable_[index] ? 1 : 0);
    }

    if(enable) emptyLS_ = 0;
    else if(ledOnExpected) return;

    iME = -1;

    for(EEDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const DetId& id(digiItr->id());

      unsigned iDCC(dccId(id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - nEBDCC);

      if(!enable_[index]) continue;
      if(rtHalf(id) != rtHalf_[index]) continue;

      if(iME != wlToME_[wavelength_[index]]){
        iME = wlToME_[wavelength_[index]];
        static_cast<MESetMulti&>(meShape).use(iME);
      }

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      for(int iSample(0); iSample < 10; iSample++)
	meShape.fill(id, iSample + 0.5, float(dataFrame.sample(iSample).adc()));

      EcalPnDiodeDetId pnidA(pnForCrystal(id, 'a'));
      EcalPnDiodeDetId pnidB(pnForCrystal(id, 'b'));
      if(pnidA.null() || pnidB.null()) continue;
      pnAmp_.insert(std::make_pair(pnidA.rawId(), 0.));
      pnAmp_.insert(std::make_pair(pnidB.rawId(), 0.));
    }
  }

  void
  LedTask::runOnPnDigis(EcalPnDiodeDigiCollection const& _digis)
  {
    MESet& mePNAmplitude(MEs_.at("PNAmplitude"));

    unsigned iME(-1);

    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      if(digiItr->sample(0).gainId() != 0 && digiItr->sample(0).gainId() != 1) continue;

      const EcalPnDiodeDetId& id(digiItr->id());

      std::map<uint32_t, float>::iterator ampItr(pnAmp_.find(id.rawId()));
      if(ampItr == pnAmp_.end()) continue;

      unsigned iDCC(dccId(id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - nEBDCC);

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
        static_cast<MESetMulti&>(mePNAmplitude).use(iME);
      }

      mePNAmplitude.fill(id, max);

      ampItr->second = max;
    }
  }

  void
  LedTask::runOnUncalibRecHits(EcalUncalibratedRecHitCollection const& _uhits)
  {
    using namespace std;

    MESet& meAmplitude(MEs_.at("Amplitude"));
    MESet& meAmplitudeSummary(MEs_.at("AmplitudeSummary"));
    MESet& meTiming(MEs_.at("Timing"));
    MESet& meAOverP(MEs_.at("AOverP"));

    unsigned iME(-1);

    for(EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr){
      EEDetId id(uhitItr->id());

      unsigned iDCC(dccId(id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - nEBDCC);

      if(!enable_[index]) continue;
      if(rtHalf(id) != rtHalf_[index]) continue;

      if(iME != wlToME_[wavelength_[index]]){
        iME = wlToME_[wavelength_[index]];
        static_cast<MESetMulti&>(meAmplitude).use(iME);
        static_cast<MESetMulti&>(meAmplitudeSummary).use(iME);
        static_cast<MESetMulti&>(meTiming).use(iME);
        static_cast<MESetMulti&>(meAOverP).use(iME);
      }

      float amp(max((double)uhitItr->amplitude(), 0.));
      float jitter(max((double)uhitItr->jitter() + 5.0, 0.));

      meAmplitude.fill(id, amp);
      meAmplitudeSummary.fill(id, amp);
      meTiming.fill(id, jitter);

      float aop(0.);

      map<uint32_t, float>::iterator ampItrA(pnAmp_.find(pnForCrystal(id, 'a')));
      map<uint32_t, float>::iterator ampItrB(pnAmp_.find(pnForCrystal(id, 'b')));
      if(ampItrA == pnAmp_.end() && ampItrB == pnAmp_.end()) continue;
      else if(ampItrB == pnAmp_.end()) aop = amp / ampItrA->second;
      else if(ampItrA == pnAmp_.end()) aop = amp / ampItrB->second;
      else aop = amp / (ampItrA->second + ampItrB->second) * 2.;

      meAOverP.fill(id, aop);
    }
  }

  DEFINE_ECALDQM_WORKER(LedTask);
}

