#include "../interface/LaserTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

namespace ecaldqm {

  LaserTask::LaserTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "LaserTask"),
    wlToME_(),
    pnAmp_(),
    emptyLS_(0),
    emptyLSLimit_(_workerParams.getUntrackedParameter<int>("emptyLSLimit"))
  {
    using namespace std;

    collectionMask_[kEcalRawData] = true;
    collectionMask_[kEBDigi] = true;
    collectionMask_[kEEDigi] = true;
    collectionMask_[kPnDiodeDigi] = true;
    collectionMask_[kEBLaserLedUncalibRecHit] = true;
    collectionMask_[kEELaserLedUncalibRecHit] = true;

    for(unsigned iD(0); iD < BinService::nDCC; ++iD){
      enable_[iD] = false;
      wavelength_[iD] = 0;
      rtHalf_[iD] = 0;
    }

    vector<int> laserWavelengths(_commonParams.getUntrackedParameter<vector<int> >("laserWavelengths"));

    unsigned iMEWL(0);
    for(vector<int>::iterator wlItr(laserWavelengths.begin()); wlItr != laserWavelengths.end(); ++wlItr){
      if(*wlItr <= 0 || *wlItr >= 5) throw cms::Exception("InvalidConfiguration") << "Laser Wavelength" << endl;
      wlToME_[*wlItr] = iMEWL++;
    }

    map<string, string> replacements;
    stringstream ss;

    std::string wlPlots[] = {"Amplitude", "AmplitudeSummary", "Timing", "Shape", "AOverP", "PNAmplitude"};
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

  }

  void
  LaserTask::setDependencies(DependencySet& _dependencies)
  {
    _dependencies.push_back(Dependency(kEBDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kEEDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kPnDiodeDigi, kEBDigi, kEEDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kEBLaserLedUncalibRecHit, kPnDiodeDigi, kEBDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kEELaserLedUncalibRecHit, kPnDiodeDigi, kEEDigi, kEcalRawData));
  }

  bool
  LaserTask::filterRunType(const std::vector<short>& _runType)
  {
    bool enable(false);

    for(unsigned iDCC(0); iDCC < BinService::nDCC; iDCC++){
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

  void
  LaserTask::beginRun(edm::Run const&, edm::EventSetup const&)
  {
    emptyLS_ = 0;
  }

  void
  LaserTask::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
  {
    if(++emptyLS_ > emptyLSLimit_) emptyLS_ = -1;
  }

  void
  LaserTask::beginEvent(const edm::Event &_evt, const edm::EventSetup &)
  {
    pnAmp_.clear();
  }

  void
  LaserTask::runOnRawData(EcalRawDataCollection const& _rawData)
  {
    for(EcalRawDataCollection::const_iterator rItr(_rawData.begin()); rItr != _rawData.end(); ++rItr){
      unsigned iDCC(rItr->id() - 1);

      if(!enable_[iDCC]){
        wavelength_[iDCC] = -1;
        rtHalf_[iDCC] = -1;
        continue;
      }
      wavelength_[iDCC] = rItr->getEventSettings().wavelength + 1;

      if(wlToME_.find(wavelength_[iDCC]) == wlToME_.end())
        enable_[iDCC] = false;

      rtHalf_[iDCC] = rItr->getRtHalf();
    }
  }

  void
  LaserTask::runOnDigis(const EcalDigiCollection &_digis)
  {
    MESet* meOccupancy(MEs_["Occupancy"]);
    MESet* meShape(MEs_["Shape"]);

    int nReadouts[BinService::nDCC];
    int maxpos[BinService::nDCC][10];
    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      nReadouts[iDCC] = 0;
      for(int i(0); i < 10; i++) maxpos[iDCC][i] = 0;
    }

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const DetId& id(digiItr->id());

      meOccupancy->fill(id);

      unsigned iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;
      if(rtHalf(id) != rtHalf_[iDCC]) continue;

      ++nReadouts[iDCC];

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
        maxpos[iDCC][iMax] += 1;
    }

    // signal existence check
    bool enable(false);
    bool laserOnExpected(emptyLS_ >= 0);

    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      if(nReadouts[iDCC] == 0){
        enable_[iDCC] = false;
        continue;
      }

      int threshold(nReadouts[iDCC] / 3);
      if(laserOnExpected) enable_[iDCC] = false;

      for(int i(0); i < 10; i++){
        if(maxpos[iDCC][i] > threshold){
          enable = true;
          enable_[iDCC] = true;
          break;
        }
      }
    }

    if(enable) emptyLS_ = 0;
    else if(laserOnExpected) return;

    unsigned iME(-1);
    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const DetId& id(digiItr->id());

      unsigned iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;
      if(rtHalf(id) != rtHalf_[iDCC]) continue;

      EcalDataFrame dataFrame(*digiItr);

      if(iME != wlToME_[wavelength_[iDCC]]){
        iME = wlToME_[wavelength_[iDCC]];
        static_cast<MESetMulti*>(meShape)->use(iME);
      }

      for(int iSample(0); iSample < 10; iSample++)
	meShape->fill(id, iSample + 0.5, float(dataFrame.sample(iSample).adc()));

      EcalPnDiodeDetId pnidA(pnForCrystal(id, 'a'));
      EcalPnDiodeDetId pnidB(pnForCrystal(id, 'b'));
      if(pnidA.null() || pnidB.null()) continue;
      pnAmp_.insert(std::make_pair(pnidA.rawId(), 0.));
      pnAmp_.insert(std::make_pair(pnidB.rawId(), 0.));
    }
  }

  void
  LaserTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
  {
    MESet* mePNAmplitude(MEs_["PNAmplitude"]);

    bool enable(false);
    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC)
      enable |= enable_[iDCC];
    if(!enable) return;

    unsigned iME(-1);

    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      if(digiItr->sample(0).gainId() != 0 && digiItr->sample(0).gainId() != 1) continue;

      const EcalPnDiodeDetId& id(digiItr->id());

      std::map<uint32_t, float>::iterator ampItr(pnAmp_.find(id.rawId()));
      if(ampItr == pnAmp_.end()) continue;

      unsigned iDCC(dccId(id) - 1);

      double pedestal(0.);
      for(int iSample(0); iSample < 4; iSample++)
	pedestal += digiItr->sample(iSample).adc();
      pedestal /= 4.;

      double max(0.);
      for(int iSample(0); iSample < 50; iSample++){
	float amp(digiItr->sample(iSample).adc() - pedestal);
	if(amp > max) max = amp;
      }

      if(iME != wlToME_[wavelength_[iDCC]]){
        iME = wlToME_[wavelength_[iDCC]];
        static_cast<MESetMulti*>(mePNAmplitude)->use(iME);
      }

      mePNAmplitude->fill(id, max);

      ampItr->second = max;
    }
  }

  void
  LaserTask::runOnUncalibRecHits(const EcalUncalibratedRecHitCollection &_uhits)
  {
    MESet* meAmplitude(MEs_["Amplitude"]);
    MESet* meAmplitudeSummary(MEs_["AmplitudeSummary"]);
    MESet* meTiming(MEs_["Timing"]);
    MESet* meAOverP(MEs_["AOverP"]);

    using namespace std;

    bool enable(false);
    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC)
      enable |= enable_[iDCC];
    if(!enable) return;

    unsigned iME(-1);

    for(EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr){
      const DetId& id(uhitItr->id());

      unsigned iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;
      if(rtHalf(id) != rtHalf_[iDCC]) continue;

      if(iME != wlToME_[wavelength_[iDCC]]){
        iME = wlToME_[wavelength_[iDCC]];
        static_cast<MESetMulti*>(meAmplitude)->use(iME);
        static_cast<MESetMulti*>(meAmplitudeSummary)->use(iME);
        static_cast<MESetMulti*>(meTiming)->use(iME);
        static_cast<MESetMulti*>(meAOverP)->use(iME);
      }

      float amp(max((double)uhitItr->amplitude(), 0.));
      float jitter(max((double)uhitItr->jitter() + 5.0, 0.));

      meAmplitude->fill(id, amp);
      meAmplitudeSummary->fill(id, amp);
      meTiming->fill(id, jitter);

      float aop(0.);

      map<uint32_t, float>::iterator ampItrA(pnAmp_.find(pnForCrystal(id, 'a')));
      map<uint32_t, float>::iterator ampItrB(pnAmp_.find(pnForCrystal(id, 'b')));
      if(ampItrA == pnAmp_.end() && ampItrB == pnAmp_.end()) continue;
      else if(ampItrB == pnAmp_.end()) aop = amp / ampItrA->second;
      else if(ampItrA == pnAmp_.end()) aop = amp / ampItrB->second;
      else aop = amp / (ampItrA->second + ampItrB->second) * 2.;

      meAOverP->fill(id, aop);
    }
  }

  DEFINE_ECALDQM_WORKER(LaserTask);
}

