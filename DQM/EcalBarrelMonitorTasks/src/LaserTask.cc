#include "../interface/LaserTask.h"

#include <cmath>

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEBGeom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEEGeom.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

namespace ecaldqm {

  LaserTask::LaserTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "LaserTask"),
    wlToME_(),
    pnAmp_()
  {
    using namespace std;

    collectionMask_ = 
      (0x1 << kEBDigi) |
      (0x1 << kEEDigi) |
      (0x1 << kPnDiodeDigi) |
      (0x1 << kEBLaserLedUncalibRecHit) |
      (0x1 << kEELaserLedUncalibRecHit);

    for(unsigned iD(0); iD < BinService::nDCC; ++iD){
      enable_[iD] = false;
      wavelength_[iD] = 0;
    }

    vector<int> laserWavelengths(_commonParams.getUntrackedParameter<vector<int> >("laserWavelengths"));

    unsigned iMEWL(0);
    for(vector<int>::iterator wlItr(laserWavelengths.begin()); wlItr != laserWavelengths.end(); ++wlItr){
      if(*wlItr <= 0 || *wlItr >= 5) throw cms::Exception("InvalidConfiguration") << "Laser Wavelength" << endl;
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
  LaserTask::beginEvent(const edm::Event &, const edm::EventSetup &)
  {
    pnAmp_.clear();
  }

  bool
  LaserTask::filterRunType(const std::vector<short>& _runType)
  {
    bool enable(false);

    for(int iDCC(0); iDCC < BinService::nDCC; iDCC++){
      if(_runType[iDCC] == EcalDCCHeaderBlock::LASER_STD ||
	 _runType[iDCC] == EcalDCCHeaderBlock::LASER_GAP){
	enable = true;
	enable_[iDCC] = true;
      }
      else
        enable_[iDCC] = false;
    }

    return enable;
  }

  bool
  LaserTask::filterEventSetting(const std::vector<EventSettings>& _setting)
  {
    bool enable(false);

    for(int iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      if(!enable_[iDCC]){
        wavelength_[iDCC] = -1;
        continue;
      }
      wavelength_[iDCC] = _setting[iDCC].wavelength + 1;

      if(wlToME_.find(wavelength_[iDCC]) != wlToME_.end())
        enable = true;
      else
        enable_[iDCC] = false;
    }

    return enable;
  }

  void
  LaserTask::runOnDigis(const EcalDigiCollection &_digis)
  {
    int nReadouts(0);
    std::vector<int> maxpos(10, 0);

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const DetId& id(digiItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      ++nReadouts;

      EcalDataFrame dataFrame(*digiItr);

      int iMax(-1);
      int max(0);
      for (int i = 0; i < 10; i++) {
        int adc = dataFrame.sample(i).adc();
        if(adc > max){
          max = adc;
          iMax = i;
        }
      }
      if(iMax >= 0)
        maxpos[iMax] += 1;
    }

    bool majorityExists(false);
    int threshold(nReadouts / 2);
    for(int i(0); i < 10; i++){
      if(maxpos[i] > threshold){
        majorityExists = true;
        break;
      }
    }

    if(!majorityExists){
      for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC)
        enable_[iDCC] = false;
      return;
    }

    unsigned iME(-1);
    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const DetId& id(digiItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      EcalDataFrame dataFrame(*digiItr);

      if(iME != wlToME_[wavelength_[iDCC]]){
        iME = wlToME_[wavelength_[iDCC]];
        static_cast<MESetMulti*>(MEs_[kOccupancy])->use(iME);
        static_cast<MESetMulti*>(MEs_[kShape])->use(iME);
      }

      MEs_[kOccupancy]->fill(id);

      for(int iSample(0); iSample < 10; iSample++)
	MEs_[kShape]->fill(id, iSample + 0.5, float(dataFrame.sample(iSample).adc()));
    }
  }

  void
  LaserTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
  {
    unsigned iME(-1);

    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const EcalPnDiodeDetId& id(digiItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      double pedestal(0.);
      for(int iSample(0); iSample < 4; iSample++)
	pedestal += digiItr->sample(iSample).adc();
      pedestal /= 4.;

      double max(0.);
      for(int iSample(0); iSample < 50; iSample++){
	EcalFEMSample sample(digiItr->sample(iSample));

	float amp(digiItr->sample(iSample).adc() - pedestal);

	if(amp > max) max = amp;
      }

      if(iME != wlToME_[wavelength_[iDCC]]){
        iME = wlToME_[wavelength_[iDCC]];
        static_cast<MESetMulti*>(MEs_[kPNAmplitude])->use(iME);
      }

      MEs_[kPNAmplitude]->fill(id, max);

      if(pnAmp_.find(iDCC) == pnAmp_.end()) pnAmp_[iDCC].resize(10);
      pnAmp_[iDCC][id.iPnId() - 1] = max;
    }
  }

  void
  LaserTask::runOnUncalibRecHits(const EcalUncalibratedRecHitCollection &_uhits, Collections _collection)
  {
    using namespace std;

    unsigned iME(-1);

    for(EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr){
      const DetId& id(uhitItr->id());

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

      if(_collection == kEBLaserLedUncalibRecHit){
	EBDetId ebid(id);

	int lmmod(MEEBGeom::lmmod(ebid.ieta(), ebid.iphi()));
	pair<int, int> pnPair(MEEBGeom::pn(lmmod));

	pn0 = pnAmp_[iDCC][pnPair.first];
	pn1 = pnAmp_[iDCC][pnPair.second];
      }
      else if(_collection == kEELaserLedUncalibRecHit){
	EcalScDetId scid(EEDetId(id).sc());

	int dee(MEEEGeom::dee(scid.ix(), scid.iy(), scid.zside()));
	int lmmod(MEEEGeom::lmmod(scid.ix(), scid.iy()));
	pair<int, int> pnPair(MEEEGeom::pn(dee, lmmod));

	int pnADCC(EEPnDCC(dee, 0) - 601), pnBDCC(EEPnDCC(dee, 1) - 601);

	pn0 = pnAmp_[pnADCC][pnPair.first];
	pn1 = pnAmp_[pnBDCC][pnPair.second];
      }

      if(pn0 < 10 && pn1 > 10) aop = amp / pn1;
      else if(pn0 > 10 && pn1 < 10) aop = amp / pn0;
      else if(pn0 + pn1 > 1) aop = amp / (0.5 * (pn0 + pn1));
      else aop = 1000.;

      MEs_[kAOverP]->fill(id, aop);
    }
  }

  /*static*/
  void
  LaserTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["AmplitudeSummary"] = kAmplitudeSummary;
    _nameToIndex["Amplitude"] = kAmplitude;
    _nameToIndex["Occupancy"] = kOccupancy;
    _nameToIndex["Timing"] = kTiming;
    _nameToIndex["Shape"] = kShape;
    _nameToIndex["AOverP"] = kAOverP;
    _nameToIndex["PNAmplitude"] = kPNAmplitude;
  }

  DEFINE_ECALDQM_WORKER(LaserTask);
}

