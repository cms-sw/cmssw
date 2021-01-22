#include "DQM/EcalMonitorTasks/interface/LaserTask.h"

#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm {
  LaserTask::LaserTask() : DQWorkerTask(), wlToME_(), pnAmp_(), emptyLS_(0), emptyLSLimit_(0), maxPedestal_(0) {
    std::fill_n(enable_, nDCC, false);
    std::fill_n(wavelength_, nDCC, 0);
    std::fill_n(rtHalf_, nDCC, 0);
  }

  void LaserTask::setParams(edm::ParameterSet const& _params) {
    emptyLSLimit_ = _params.getUntrackedParameter<int>("emptyLSLimit");

    std::vector<int> laserWavelengths(_params.getUntrackedParameter<std::vector<int> >("laserWavelengths"));

    MESet::PathReplacements repl;

    MESetMulti& amplitude(static_cast<MESetMulti&>(MEs_.at("Amplitude")));
    unsigned nWL(laserWavelengths.size());
    for (unsigned iWL(0); iWL != nWL; ++iWL) {
      int wl(laserWavelengths[iWL]);
      if (wl <= 0 || wl >= 5)
        throw cms::Exception("InvalidConfiguration") << "Laser Wavelength";
      repl["wl"] = std::to_string(wl);
      wlToME_[wl] = amplitude.getIndex(repl);
    }

    maxPedestal_ = _params.getUntrackedParameter<int>("maxPedestal");
  }

  void LaserTask::addDependencies(DependencySet& _dependencies) {
    _dependencies.push_back(Dependency(kEBDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kEEDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kPnDiodeDigi, kEBDigi, kEEDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kEBLaserLedUncalibRecHit, kPnDiodeDigi, kEBDigi, kEcalRawData));
    _dependencies.push_back(Dependency(kEELaserLedUncalibRecHit, kPnDiodeDigi, kEEDigi, kEcalRawData));
  }

  bool LaserTask::filterRunType(short const* _runType) {
    bool enable(false);

    for (unsigned iDCC(0); iDCC < nDCC; iDCC++) {
      if (_runType[iDCC] == EcalDCCHeaderBlock::LASER_STD || _runType[iDCC] == EcalDCCHeaderBlock::LASER_GAP) {
        enable = true;
        enable_[iDCC] = true;
      } else
        enable_[iDCC] = false;
    }

    return enable;
  }

  void LaserTask::beginRun(edm::Run const&, edm::EventSetup const&) { emptyLS_ = 0; }

  void LaserTask::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
    if (++emptyLS_ > emptyLSLimit_)
      emptyLS_ = -1;
  }

  void LaserTask::beginEvent(edm::Event const& _evt, edm::EventSetup const&, bool const&, bool&) { pnAmp_.clear(); }

  void LaserTask::runOnRawData(EcalRawDataCollection const& _rawData) {
    MESet& meCalibStatus(MEs_.at("CalibStatus"));
    for (EcalRawDataCollection::const_iterator rItr(_rawData.begin()); rItr != _rawData.end(); ++rItr) {
      unsigned iDCC(rItr->id() - 1);

      if (!enable_[iDCC]) {
        wavelength_[iDCC] = -1;
        rtHalf_[iDCC] = -1;
        continue;
      }
      wavelength_[iDCC] = rItr->getEventSettings().wavelength + 1;

      if (wlToME_.find(wavelength_[iDCC]) == wlToME_.end())
        enable_[iDCC] = false;

      rtHalf_[iDCC] = rItr->getRtHalf();
    }
    bool LaserStatus[nWavelength];
    for (unsigned iW(0); iW < nWavelength; iW++) {
      LaserStatus[iW] = false;
    }
    for (unsigned iDCC(0); iDCC < nDCC; iDCC++) {
      switch (wavelength_[iDCC]) {
        case 1:
          break;
        case 2:
          LaserStatus[Wavelength::kGreen] = true;
          break;
        case 3:
          LaserStatus[Wavelength::kBlue] = true;
          break;
        case 4:
          LaserStatus[Wavelength::kIRed] = true;
          break;
        default:
          break;
      }
    }
    for (unsigned iWL(0); iWL < nWavelength; iWL++) {
      meCalibStatus.fill(double(iWL), LaserStatus[iWL] ? 1 : 0);
    }
  }

  template <typename DigiCollection>
  void LaserTask::runOnDigis(DigiCollection const& _digis) {
    MESet& meOccupancy(MEs_.at("Occupancy"));
    MESet& meShape(MEs_.at("Shape"));
    MESet& meSignalRate(MEs_.at("SignalRate"));

    bool inData[nDCC];
    int nReadouts[nDCC];
    int maxpos[nDCC][EcalDataFrame::MAXSAMPLES];
    bool largeAmplitude[nDCC];
    for (unsigned iDCC(0); iDCC < nDCC; ++iDCC) {
      inData[iDCC] = false;
      nReadouts[iDCC] = 0;
      for (int i(0); i < EcalDataFrame::MAXSAMPLES; i++)
        maxpos[iDCC][i] = 0;
      largeAmplitude[iDCC] = false;
    }

    for (typename DigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr) {
      const DetId& id(digiItr->id());

      unsigned iDCC(dccId(id) - 1);

      inData[iDCC] = true;

      if (!enable_[iDCC])
        continue;
      if (rtHalf(id) != rtHalf_[iDCC])
        continue;

      meOccupancy.fill(id);

      ++nReadouts[iDCC];

      EcalDataFrame dataFrame(*digiItr);

      int iMax(-1);
      int max(0);
      int min(4096);
      for (int i(0); i < EcalDataFrame::MAXSAMPLES; i++) {
        int adc(dataFrame.sample(i).adc());
        if (adc > max) {
          max = adc;
          iMax = i;
        }
        if (adc < min)
          min = adc;
        if (adc > maxPedestal_)
          largeAmplitude[iDCC] = true;
      }
      if (iMax >= 0 && max - min > 3)  // normal RMS of pedestal is ~2.5
        maxpos[iDCC][iMax] += 1;
    }

    // signal existence check
    bool enable(false);
    bool laserOnExpected(emptyLS_ >= 0);

    unsigned iME(-1);

    for (int iDCC(0); iDCC < nDCC; ++iDCC) {
      if (!inData[iDCC])
        continue;

      if (nReadouts[iDCC] == 0) {
        enable_[iDCC] = false;
        continue;
      }

      int threshold(nReadouts[iDCC] / 3);
      if (laserOnExpected)
        enable_[iDCC] = false;

      if (largeAmplitude[iDCC]) {
        enable = true;
        enable_[iDCC] = true;
      } else {
        for (int i(0); i < EcalDataFrame::MAXSAMPLES; i++) {
          if (maxpos[iDCC][i] > threshold) {
            enable = true;
            enable_[iDCC] = true;
            break;
          }
        }
      }

      if (iME != wlToME_[wavelength_[iDCC]]) {
        iME = wlToME_[wavelength_[iDCC]];
        static_cast<MESetMulti&>(meSignalRate).use(iME);
      }

      meSignalRate.fill(iDCC + 1, enable_[iDCC] ? 1 : 0);
    }

    if (enable)
      emptyLS_ = 0;
    else if (laserOnExpected)
      return;

    iME = -1;

    for (typename DigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr) {
      const DetId& id(digiItr->id());

      unsigned iDCC(dccId(id) - 1);

      if (!enable_[iDCC])
        continue;
      if (rtHalf(id) != rtHalf_[iDCC])
        continue;

      EcalDataFrame dataFrame(*digiItr);

      if (iME != wlToME_[wavelength_[iDCC]]) {
        iME = wlToME_[wavelength_[iDCC]];
        static_cast<MESetMulti&>(meShape).use(iME);
      }

      for (int iSample(0); iSample < EcalDataFrame::MAXSAMPLES; iSample++)
        meShape.fill(id, iSample + 0.5, float(dataFrame.sample(iSample).adc()));

      EcalPnDiodeDetId pnidA(pnForCrystal(id, 'a'));
      EcalPnDiodeDetId pnidB(pnForCrystal(id, 'b'));
      if (pnidA.null() || pnidB.null())
        continue;
      pnAmp_.insert(std::make_pair(pnidA.rawId(), 0.));
      pnAmp_.insert(std::make_pair(pnidB.rawId(), 0.));
    }
  }

  void LaserTask::runOnPnDigis(EcalPnDiodeDigiCollection const& _digis) {
    MESet& mePNAmplitude(MEs_.at("PNAmplitude"));

    bool enable(false);
    for (unsigned iDCC(0); iDCC < nDCC; ++iDCC)
      enable |= enable_[iDCC];
    if (!enable)
      return;

    unsigned iME(-1);

    for (EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr) {
      if (digiItr->sample(0).gainId() != 0 && digiItr->sample(0).gainId() != 1)
        continue;

      const EcalPnDiodeDetId& id(digiItr->id());

      std::map<uint32_t, float>::iterator ampItr(pnAmp_.find(id.rawId()));
      if (ampItr == pnAmp_.end())
        continue;

      unsigned iDCC(dccId(id) - 1);

      double pedestal(0.);
      for (int iSample(0); iSample < 4; iSample++)
        pedestal += digiItr->sample(iSample).adc();
      pedestal /= 4.;

      double max(0.);
      for (int iSample(0); iSample < 50; iSample++) {
        float amp(digiItr->sample(iSample).adc() - pedestal);
        if (amp > max)
          max = amp;
      }

      if (iME != wlToME_[wavelength_[iDCC]]) {
        iME = wlToME_[wavelength_[iDCC]];
        static_cast<MESetMulti&>(mePNAmplitude).use(iME);
      }

      mePNAmplitude.fill(id, max);

      ampItr->second = max;
    }
  }

  void LaserTask::runOnUncalibRecHits(EcalUncalibratedRecHitCollection const& _uhits) {
    MESet& meAmplitude(MEs_.at("Amplitude"));
    MESet& meAmplitudeSummary(MEs_.at("AmplitudeSummary"));
    MESet& meTiming(MEs_.at("Timing"));
    MESet& meAOverP(MEs_.at("AOverP"));

    using namespace std;

    bool enable(false);
    for (unsigned iDCC(0); iDCC < nDCC; ++iDCC)
      enable |= enable_[iDCC];
    if (!enable)
      return;

    unsigned iME(-1);

    for (EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr) {
      const DetId& id(uhitItr->id());

      unsigned iDCC(dccId(id) - 1);

      if (!enable_[iDCC])
        continue;
      if (rtHalf(id) != rtHalf_[iDCC])
        continue;

      if (iME != wlToME_[wavelength_[iDCC]]) {
        iME = wlToME_[wavelength_[iDCC]];
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
      if (ampItrA == pnAmp_.end() && ampItrB == pnAmp_.end())
        continue;
      else if (ampItrB == pnAmp_.end())
        aop = amp / ampItrA->second;
      else if (ampItrA == pnAmp_.end())
        aop = amp / ampItrB->second;
      else
        aop = amp / (ampItrA->second + ampItrB->second) * 2.;

      meAOverP.fill(id, aop);
    }
  }

  DEFINE_ECALDQM_WORKER(LaserTask);
}  // namespace ecaldqm
