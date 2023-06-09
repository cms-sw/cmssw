#include "DQM/EcalMonitorTasks/interface/TimingTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm {
  TimingTask::TimingTask()
      : DQWorkerTask(),
        bxBinEdges_(),
        bxBin_(0.),
        chi2ThresholdEB_(0.),
        chi2ThresholdEE_(0.),
        energyThresholdEB_(0.),
        energyThresholdEE_(0.),
        energyThresholdEEFwd_(0.),
        timingVsBXThreshold_(0.),
        timeErrorThreshold_(0.),
        meTimeMapByLS(nullptr) {}

  void TimingTask::setParams(edm::ParameterSet const& _params) {
    bxBinEdges_ = onlineMode_ ? _params.getUntrackedParameter<std::vector<int> >("bxBins")
                              : _params.getUntrackedParameter<std::vector<int> >("bxBinsFine");
    chi2ThresholdEB_ = _params.getUntrackedParameter<double>("chi2ThresholdEB");
    chi2ThresholdEE_ = _params.getUntrackedParameter<double>("chi2ThresholdEE");
    energyThresholdEB_ = _params.getUntrackedParameter<double>("energyThresholdEB");
    energyThresholdEE_ = _params.getUntrackedParameter<double>("energyThresholdEE");
    energyThresholdEEFwd_ = _params.getUntrackedParameter<double>("energyThresholdEEFwd");
    timingVsBXThreshold_ = _params.getUntrackedParameter<double>("timingVsBXThreshold");
    timeErrorThreshold_ = _params.getUntrackedParameter<double>("timeErrorThreshold");
    splashSwitch_ = _params.getUntrackedParameter<bool>("splashSwitch", false);
  }

  bool TimingTask::filterRunType(short const* _runType) {
    for (int iFED(0); iFED < nDCC; iFED++) {
      if (_runType[iFED] == EcalDCCHeaderBlock::COSMIC || _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
          _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
          _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL || _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
          _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL)
        return true;
    }

    return false;
  }

  void TimingTask::beginEvent(edm::Event const& _evt, edm::EventSetup const& _es, bool const& ByLumiResetSwitch, bool&) {
    using namespace std;
    std::vector<int>::iterator pBin = std::upper_bound(bxBinEdges_.begin(), bxBinEdges_.end(), _evt.bunchCrossing());
    bxBin_ = static_cast<int>(pBin - bxBinEdges_.begin()) - 0.5;
    if (ByLumiResetSwitch) {
      meTimeMapByLS = &MEs_.at("TimeMapByLS");
      if (timestamp_.iLumi % 10 == 0)
        meTimeMapByLS->reset(GetElectronicsMap());
    }
  }

  void TimingTask::runOnRecHits(EcalRecHitCollection const& _hits, Collections _collection) {
    MESet& meTimeAmp(MEs_.at("TimeAmp"));
    MESet& meTimeAmpAll(MEs_.at("TimeAmpAll"));
    MESet& meTimingVsBX(onlineMode_ ? MEs_.at("BarrelTimingVsBX") : MEs_.at("BarrelTimingVsBXFineBinned"));
    MESet& meTimeAll(MEs_.at("TimeAll"));
    MESet& meTimeAllMap(MEs_.at("TimeAllMap"));
    MESet& meTimeMap(MEs_.at("TimeMap"));  // contains cumulative run stats => not suitable for Trend plots
    MESet& meTime1D(MEs_.at("Time1D"));
    MESet& meChi2(MEs_.at("Chi2"));

    uint32_t goodOROOTBits(0x1 << EcalRecHit::kGood | 0x1 << EcalRecHit::kOutOfTime);
    int signedSubdet;

    std::for_each(_hits.begin(), _hits.end(), [&](EcalRecHitCollection::value_type const& hit) {
      if (!hit.checkFlagMask(goodOROOTBits))
        return;

      DetId id(hit.id());

      float time(hit.time());
      float energy(hit.energy());

      float energyThreshold;
      if (id.subdetId() == EcalBarrel) {
        energyThreshold = energyThresholdEB_;
        signedSubdet = EcalBarrel;
      } else {
        energyThreshold = (isForward(id)) ? energyThresholdEEFwd_ : energyThresholdEE_;
        EEDetId eeId(hit.id());
        if (eeId.zside() < 0)
          signedSubdet = -EcalEndcap;
        else
          signedSubdet = EcalEndcap;
      }

      if (energy > energyThreshold)
        meChi2.fill(getEcalDQMSetupObjects(), signedSubdet, hit.chi2());

      if (!splashSwitch_) {  //Not applied for splash events
        float chi2Threshold;
        if (id.subdetId() == EcalBarrel)
          chi2Threshold = chi2ThresholdEB_;
        else
          chi2Threshold = chi2ThresholdEE_;

        //Apply cut on chi2 of pulse shape fit
        if (hit.chi2() > chi2Threshold)
          return;
      }
      // Apply cut based on timing error of rechit
      if (hit.timeError() > timeErrorThreshold_)
        return;

      meTimeAmp.fill(getEcalDQMSetupObjects(), id, energy, time);
      meTimeAmpAll.fill(getEcalDQMSetupObjects(), id, energy, time);

      if (energy > timingVsBXThreshold_ && signedSubdet == EcalBarrel)
        meTimingVsBX.fill(getEcalDQMSetupObjects(), bxBin_, time);

      if (energy > energyThreshold) {
        meTimeAll.fill(getEcalDQMSetupObjects(), id, time);
        meTimeMap.fill(getEcalDQMSetupObjects(), id, time);
        meTimeMapByLS->fill(getEcalDQMSetupObjects(), id, time);
        meTime1D.fill(getEcalDQMSetupObjects(), id, time);
        meTimeAllMap.fill(getEcalDQMSetupObjects(), id, time);
      }
    });
  }

  // For In-time vs Out-of-Time amplitude correlation MEs:
  // Only UncalibRecHits carry information about OOT amplitude
  // But still need to make sure we apply similar cuts as on RecHits
  void TimingTask::runOnUncalibRecHits(EcalUncalibratedRecHitCollection const& _uhits) {
    MESet& meTimeAmpBXm(MEs_.at("TimeAmpBXm"));
    MESet& meTimeAmpBXp(MEs_.at("TimeAmpBXp"));

    for (EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr) {
      // Apply reconstruction quality cuts
      if (!uhitItr->checkFlag(EcalUncalibratedRecHit::kGood))
        continue;
      DetId id(uhitItr->id());
      float chi2Threshold = 0.;
      float ampThreshold = 0.;
      if (id.subdetId() == EcalBarrel) {
        chi2Threshold = chi2ThresholdEB_;
        ampThreshold = 20. * energyThresholdEB_;  // 1 GeV ~ 20 ADC in EB
      } else {
        chi2Threshold = chi2ThresholdEE_;
        ampThreshold = 5. * ((isForward(id)) ? energyThresholdEEFwd_ : energyThresholdEE_);  // 1 GeV ~ 5 ADC in EE
      }

      if (uhitItr->chi2() > chi2Threshold)
        continue;

      // Apply amplitude cut based on approx rechit energy
      float amp(uhitItr->amplitude());
      if (amp < ampThreshold)
        continue;

      // Apply jitter timing cut based on approx rechit timing
      float timeOff(id.subdetId() == EcalBarrel ? 0.4 : 1.8);
      float hitTime(uhitItr->jitter() * 25. + timeOff);  // 1 jitter ~ 25 ns
      if (std::abs(hitTime) >= 5.)
        continue;

      // Fill MEs
      meTimeAmpBXm.fill(getEcalDQMSetupObjects(), id, amp, uhitItr->outOfTimeAmplitude(4));  // BX-1
      meTimeAmpBXp.fill(getEcalDQMSetupObjects(), id, amp, uhitItr->outOfTimeAmplitude(6));  // BX+1
    }
  }

  DEFINE_ECALDQM_WORKER(TimingTask);
}  // namespace ecaldqm
