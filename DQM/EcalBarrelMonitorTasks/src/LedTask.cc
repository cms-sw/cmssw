#include "../interface/LedTask.h"

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEBGeom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEEGeom.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

namespace ecaldqm {

  LedTask::LedTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "LedTask"),
    wlToME_(),
    pnAmp_()
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

    unsigned wlPlots[] = {kAmplitudeSummary, kAmplitude, kOccupancy, kTiming, kShape, kAOverP, kPNAmplitude};
    for(unsigned iS(0); iS < sizeof(wlPlots) / sizeof(unsigned); ++iS){
      unsigned plot(wlPlots[iS]);
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
  }

  void
  LedTask::setDependencies(DependencySet& _dependencies)
  {
    _dependencies.push_back(Dependency(kEEDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kPnDiodeDigi, kEEDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kEELaserLedUncalibRecHit, kPnDiodeDigi, kEEDigi, kEcalRawData));
  }

  void
  LedTask::beginEvent(const edm::Event &, const edm::EventSetup &)
  {
    pnAmp_.clear();
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

      unsigned iDCC(dccId(id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - BinService::nEBDCC);

      if(!enable_[index]) continue;
      if(rtHalf(id) != rtHalf_[index]) continue;

      ++nReadouts[index];

      EcalDataFrame dataFrame(*digiItr);

      int iMax(-1);
      int max(0);
      for (int i(0); i < 10; i++) {
        int adc(dataFrame.sample(i).adc());
        if(adc > max){
          max = adc;
          iMax = i;
        }
      }
      if(iMax >= 0)
        maxpos[index][iMax] += 1;
    }

    bool enable(false);
    for(unsigned index(0); index < BinService::nEEDCC; ++index){
      if(nReadouts[index] == 0) continue;
      int threshold(nReadouts[index] / 3);
      for(int i(0); i < 10; i++){
        if(maxpos[index][i] > threshold){
          enable_[index] = true;
          enable = true;
          break;
        }
      }
    }

    if(!enable) return;

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
        static_cast<MESetMulti*>(MEs_[kOccupancy])->use(iME);
        static_cast<MESetMulti*>(MEs_[kShape])->use(iME);
      }

      MEs_[kOccupancy]->fill(id);

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      for(int iSample(0); iSample < 10; iSample++)
	MEs_[kShape]->fill(id, iSample + 0.5, float(dataFrame.sample(iSample).adc()));

      // which PNs are used in this event?

      EcalScDetId scid(EEDetId(id).sc());

      int dee(MEEEGeom::dee(scid.ix(), scid.iy(), scid.zside()));
      int lmmod(MEEEGeom::lmmod(scid.ix(), scid.iy()));
      std::pair<int, int> pnPair(MEEEGeom::pn(dee, lmmod));
      if(dee == 1 || dee == 4){
        // PN numbers are transposed for far-side dees
        pnPair.first = (pnPair.first % 5) + 5 * (1 - pnPair.first / 5);
        pnPair.second = (pnPair.second % 5) + 5 * (1 - pnPair.second / 5);
      }
      
      unsigned pnADCC(EEPnDCC(dee, 0)), pnBDCC(EEPnDCC(dee, 1));
      if(pnAmp_.find(pnADCC) == pnAmp_.end()) pnAmp_[pnADCC].resize(10, -1.);
      if(pnAmp_.find(pnBDCC) == pnAmp_.end()) pnAmp_[pnBDCC].resize(10, -1.);
      pnAmp_[pnADCC][pnPair.first] = 0.;
      pnAmp_[pnBDCC][pnPair.second] = 0.;
    }
  }

  void
  LedTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
  {
    unsigned iME(-1);

    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      if(digiItr->sample(0).gainId() != 0 && digiItr->sample(0).gainId() != 1) continue;

      const EcalPnDiodeDetId& id(digiItr->id());

      unsigned iDCC(dccId(id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) continue;
      unsigned index(iDCC <= kEEmHigh ? iDCC : iDCC - BinService::nEBDCC);

      if(pnAmp_.find(iDCC + 1) == pnAmp_.end() || pnAmp_[iDCC + 1][id.iPnId() - 1] < 0.) continue;

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

      if(pnAmp_.find(iDCC + 1) == pnAmp_.end()) pnAmp_[iDCC + 1].resize(10);
      pnAmp_[iDCC + 1][id.iPnId() - 1] = max;
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

      MEs_[kAmplitudeSummary]->fill(id, amp);
      MEs_[kAmplitude]->fill(id, amp);
      MEs_[kTiming]->fill(id, jitter);

      float aop(0.);
      float pn0(0.), pn1(0.);

      EcalScDetId scid(id.sc());

      int dee(MEEEGeom::dee(scid.ix(), scid.iy(), scid.zside()));
      int lmmod(MEEEGeom::lmmod(scid.ix(), scid.iy()));
      pair<int, int> pnPair(MEEEGeom::pn(dee, lmmod));
      if(dee == 1 || dee == 4){
        // PN numbers are transposed for far-side dees
        pnPair.first = (pnPair.first % 5) + 5 * (1 - pnPair.first / 5);
        pnPair.second = (pnPair.second % 5) + 5 * (1 - pnPair.second / 5);
      }

      unsigned pnAFED(EEPnDCC(dee, 0)), pnBFED(EEPnDCC(dee, 1));
      if(pnAmp_.find(pnAFED) == pnAmp_.end() || pnAmp_[pnAFED][pnPair.first] < 0.) continue;
      if(pnAmp_.find(pnBFED) == pnAmp_.end() || pnAmp_[pnBFED][pnPair.second] < 0.) continue;

      pn0 = pnAmp_[pnAFED][pnPair.first];
      pn1 = pnAmp_[pnBFED][pnPair.second];

      if(pn0 < 10 && pn1 > 10){
	aop = amp / pn1;
      }else if(pn0 > 10 && pn1 < 10){
	aop = amp / pn0;
      }else if(pn0 + pn1 > 1){
	aop = amp / (0.5 * (pn0 + pn1));
      }else{
	aop = 1000.;
      }

      MEs_[kAOverP]->fill(id, aop);
    }
  }

  /*static*/
  void
  LedTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["AmplitudeSummary"] = kAmplitudeSummary;
    _nameToIndex["Amplitude"] = kAmplitude;
    _nameToIndex["Occupancy"] = kOccupancy;
    _nameToIndex["Shape"] = kShape;
    _nameToIndex["Timing"] = kTiming;
    _nameToIndex["AOverP"] = kAOverP;
    _nameToIndex["PNAmplitude"] = kPNAmplitude;
  }

  DEFINE_ECALDQM_WORKER(LedTask);
}

