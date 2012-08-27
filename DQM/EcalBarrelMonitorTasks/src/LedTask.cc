#include "../interface/LedTask.h"

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEBGeom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEEGeom.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

namespace ecaldqm {

  LedTask::LedTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "LedTask"),
    wlToME_(),
    wlGainToME_(),
    pnAmp_()
  {
    using namespace std;

    collectionMask_ = 
      (0x1 << kEEDigi) |
      (0x1 << kPnDiodeDigi) |
      (0x1 << kEBUncalibRecHit) |
      (0x1 << kEEUncalibRecHit);

    for(unsigned iD(0); iD < BinService::nDCC; ++iD){
      enable_[iD] = false;
      wavelength_[iD] = 0;
    }

    vector<int> MGPAGainsPN(_commonParams.getUntrackedParameter<vector<int> >("MGPAGainsPN"));
    vector<int> ledWavelengths(_commonParams.getUntrackedParameter<vector<int> >("ledWavelengths"));

    unsigned iMEWL(0);
    unsigned iMEWLG(0);
    for(vector<int>::iterator wlItr(ledWavelengths.begin()); wlItr != ledWavelengths.end(); ++wlItr){
      if(*wlItr <= 0 || *wlItr >= 2) throw cms::Exception("InvalidConfiguration") << "Led Wavelength" << endl;
      wlToME_[*wlItr] = iMEWL++;

      for(vector<int>::iterator gainItr(MGPAGainsPN.begin()); gainItr != MGPAGainsPN.end(); ++gainItr){
        if(*gainItr != 1 && *gainItr != 16) throw cms::Exception("InvalidConfiguration") << "PN diode gain" << endl;
        wlGainToME_[make_pair(*wlItr, *gainItr)] = iMEWLG++;
      }
    }

    map<string, string> replacements;
    stringstream ss;

    unsigned apdPlots[] = {kAmplitudeSummary, kAmplitude, kOccupancy, kTiming, kShape, kAOverP};
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

    unsigned pnPlots[] = {kPNAmplitude};
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
  }

  LedTask::~LedTask()
  {
  }

  void
  LedTask::beginRun(const edm::Run &, const edm::EventSetup &_es)
  {
    for(int iDCC(0); iDCC < BinService::nDCC; iDCC++){
      enable_[iDCC] = false;
      wavelength_[iDCC] = -1;
    }
    pnAmp_.clear();
  }

  void
  LedTask::endEvent(const edm::Event &, const edm::EventSetup &)
  {
    for(int iDCC(0); iDCC < BinService::nDCC; iDCC++){
      enable_[iDCC] = false;
      wavelength_[iDCC] = -1;
    }
    pnAmp_.clear();
  }

  bool
  LedTask::filterRunType(const std::vector<short>& _runType)
  {
    bool enable(false);

    for(int iDCC(0); iDCC < BinService::nDCC; iDCC++){
      if(_runType[iDCC] == EcalDCCHeaderBlock::LED_STD ||
	 _runType[iDCC] == EcalDCCHeaderBlock::LED_GAP){
	enable = true;
	enable_[iDCC] = true;
      }
    }

    return enable;
  }

  bool
  LedTask::filterEventSetting(const std::vector<EventSettings>& _setting)
  {
    bool enable(false);

    for(int iDCC(0); iDCC < BinService::nDCC; iDCC++){
      if(!enable_[iDCC]) continue;
      wavelength_[iDCC] = _setting[iDCC].wavelength + 1;

      if(wlToME_.find(wavelength_[iDCC]) != wlToME_.end())
        enable = true;
      else
        enable_[iDCC] = false;
    }

    return enable;
  }

  void
  LedTask::runOnDigis(const EcalDigiCollection &_digis)
  {
    unsigned iME(-1);

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const DetId& id(digiItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      if(iME != wlToME_[wavelength_[iDCC]]){
        iME = wlToME_[wavelength_[iDCC]];
        static_cast<MESetMulti*>(MEs_[kOccupancy])->use(iME);
        static_cast<MESetMulti*>(MEs_[kShape])->use(iME);
      }

      MEs_[kOccupancy]->fill(id);

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      for(int iSample(0); iSample < 10; iSample++)
	MEs_[kShape]->fill(id, iSample + 0.5, float(dataFrame.sample(iSample).adc()));
    }
  }

  void
  LedTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
  {
    unsigned iME(-1);

    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const EcalPnDiodeDetId& id(digiItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      MEs_[kPNOccupancy]->fill(id);

      float pedestal(0.);
      for(int iSample(0); iSample < 4; iSample++)
	pedestal += digiItr->sample(iSample).adc();
      pedestal /= 4.;

      float max(0.);
      for(int iSample(0); iSample < 50; iSample++){
	EcalFEMSample sample(digiItr->sample(iSample));

	float amp(digiItr->sample(iSample).adc() - pedestal);

	if(amp > max) max = amp;
      }

      int gain(digiItr->sample(0).gainId() == 0 ? 1 : 16);
      max *= (16. / gain);

      std::pair<int, int> wlGain(wavelength_[iDCC], gain);

      if(iME != wlGainToME_[wlGain]){
        iME = wlGainToME_[wlGain];
        static_cast<MESetMulti*>(MEs_[kPNAmplitude])->use(iME);
      }

      MEs_[kPNAmplitude]->fill(id, max);

      if(pnAmp_.find(iDCC) == pnAmp_.end()) pnAmp_[iDCC].resize(10);
      pnAmp_[iDCC][id.iPnId() - 1] = max;
    }
  }

  void
  LedTask::runOnUncalibRecHits(const EcalUncalibratedRecHitCollection &_uhits)
  {
    using namespace std;

    unsigned iME(-1);

    for(EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr){
      EEDetId id(uhitItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      if(iME != wlToME_[wavelength_[iDCC]]){
        iME = wlToME_[wavelength_[iDCC]];
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

      if(pnAmp_.find(iDCC) == pnAmp_.end()) continue;

      float aop(0.);
      float pn0(0.), pn1(0.);

      EcalScDetId scid(id.sc());

      int dee(MEEEGeom::dee(scid.ix(), scid.iy(), scid.zside()));
      int lmmod(MEEEGeom::lmmod(scid.ix(), scid.iy()));
      pair<int, int> pnPair(MEEEGeom::pn(dee, lmmod));

      int pnAFED(EEPnDCC(dee, 0)), pnBFED(EEPnDCC(dee, 1));

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
    _nameToIndex["PNOccupancy"] = kPNOccupancy;
  }

  DEFINE_ECALDQM_WORKER(LedTask);
}

