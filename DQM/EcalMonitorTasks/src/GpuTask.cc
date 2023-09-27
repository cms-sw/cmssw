#include "DQM/EcalMonitorTasks/interface/GpuTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"

namespace ecaldqm {
  GpuTask::GpuTask()
      : DQWorkerTask(),
        runGpuTask_(false),
        enableDigi_(false),
        enableUncalib_(false),
        enableRecHit_(false),
        digi1D_(false),
        digi2D_(false),
        uncalib1D_(false),
        uncalib2D_(false),
        rechit1D_(false),
        rechit2D_(false),
        EBCpuDigis_(nullptr),
        EECpuDigis_(nullptr),
        EBCpuUncalibRecHits_(nullptr),
        EECpuUncalibRecHits_(nullptr),
        EBCpuRecHits_(nullptr),
        EECpuRecHits_(nullptr) {}

  void GpuTask::addDependencies(DependencySet& dependencies) {
    // Ensure we run on CPU objects before GPU objects
    if (enableDigi_) {
      dependencies.push_back(Dependency(kEBGpuDigi, kEBCpuDigi));
      dependencies.push_back(Dependency(kEEGpuDigi, kEECpuDigi));
    }

    if (enableUncalib_) {
      dependencies.push_back(Dependency(kEBGpuUncalibRecHit, kEBCpuUncalibRecHit));
      dependencies.push_back(Dependency(kEEGpuUncalibRecHit, kEECpuUncalibRecHit));
    }

    if (enableRecHit_) {
      dependencies.push_back(Dependency(kEBGpuRecHit, kEBCpuRecHit));
      dependencies.push_back(Dependency(kEEGpuRecHit, kEECpuRecHit));
    }
  }

  void GpuTask::setParams(edm::ParameterSet const& params) {
    runGpuTask_ = params.getUntrackedParameter<bool>("runGpuTask");

    // Enabling objects set to false if runGpuTask_ is false
    enableDigi_ = runGpuTask_ && params.getUntrackedParameter<bool>("enableDigi");
    enableUncalib_ = runGpuTask_ && params.getUntrackedParameter<bool>("enableUncalib");
    enableRecHit_ = runGpuTask_ && params.getUntrackedParameter<bool>("enableRecHit");

    // Flags set to false if corresponding type is not enabled
    digi1D_ = enableDigi_ && params.getUntrackedParameter<bool>("digi1D");
    digi2D_ = enableDigi_ && params.getUntrackedParameter<bool>("digi2D");
    uncalib1D_ = enableUncalib_ && params.getUntrackedParameter<bool>("uncalib1D");
    uncalib2D_ = enableUncalib_ && params.getUntrackedParameter<bool>("uncalib2D");
    rechit1D_ = enableRecHit_ && params.getUntrackedParameter<bool>("rechit1D");
    rechit2D_ = enableRecHit_ && params.getUntrackedParameter<bool>("rechit2D");

    uncalibOOTAmps_ = params.getUntrackedParameter<std::vector<int> >("uncalibOOTAmps");

    if (!enableDigi_) {
      MEs_.erase(std::string("DigiCpu"));
      MEs_.erase(std::string("DigiCpuAmplitude"));
      MEs_.erase(std::string("DigiGpuCpu"));
      MEs_.erase(std::string("DigiGpuCpuAmplitude"));
    }
    if (!enableUncalib_) {
      MEs_.erase(std::string("UncalibCpu"));
      MEs_.erase(std::string("UncalibCpuAmp"));
      MEs_.erase(std::string("UncalibCpuAmpError"));
      MEs_.erase(std::string("UncalibCpuPedestal"));
      MEs_.erase(std::string("UncalibCpuJitter"));
      MEs_.erase(std::string("UncalibCpuJitterError"));
      MEs_.erase(std::string("UncalibCpuChi2"));
      MEs_.erase(std::string("UncalibCpuOOTAmp"));
      MEs_.erase(std::string("UncalibCpuFlags"));
      MEs_.erase(std::string("UncalibGpuCpu"));
      MEs_.erase(std::string("UncalibGpuCpuAmp"));
      MEs_.erase(std::string("UncalibGpuCpuAmpError"));
      MEs_.erase(std::string("UncalibGpuCpuPedestal"));
      MEs_.erase(std::string("UncalibGpuCpuJitter"));
      MEs_.erase(std::string("UncalibGpuCpuJitterError"));
      MEs_.erase(std::string("UncalibGpuCpuChi2"));
      MEs_.erase(std::string("UncalibGpuCpuOOTAmp"));
      MEs_.erase(std::string("UncalibGpuCpuFlags"));
    }
    if (!enableRecHit_) {
      MEs_.erase(std::string("RecHitCpu"));
      MEs_.erase(std::string("RecHitCpuEnergy"));
      MEs_.erase(std::string("RecHitCpuTime"));
      MEs_.erase(std::string("RecHitCpuFlags"));
      MEs_.erase(std::string("RecHitGpuCpu"));
      MEs_.erase(std::string("RecHitGpuCpuEnergy"));
      MEs_.erase(std::string("RecHitGpuCpuTime"));
      MEs_.erase(std::string("RecHitGpuCpuFlags"));
    }
    if (!digi1D_) {
      MEs_.erase(std::string("DigiGpu"));
      MEs_.erase(std::string("DigiGpuAmplitude"));
    }
    if (!uncalib1D_) {
      MEs_.erase(std::string("UncalibGpu"));
      MEs_.erase(std::string("UncalibGpuAmp"));
      MEs_.erase(std::string("UncalibGpuAmpError"));
      MEs_.erase(std::string("UncalibGpuPedestal"));
      MEs_.erase(std::string("UncalibGpuJitter"));
      MEs_.erase(std::string("UncalibGpuJitterError"));
      MEs_.erase(std::string("UncalibGpuChi2"));
      MEs_.erase(std::string("UncalibGpuOOTAmp"));
      MEs_.erase(std::string("UncalibGpuFlags"));
    }
    if (!rechit1D_) {
      MEs_.erase(std::string("RecHitGpu"));
      MEs_.erase(std::string("RecHitGpuEnergy"));
      MEs_.erase(std::string("RecHitGpuTime"));
      MEs_.erase(std::string("RecHitGpuFlags"));
    }
    if (!digi2D_) {
      MEs_.erase(std::string("Digi2D"));
      MEs_.erase(std::string("Digi2DAmplitude"));
    }
    if (!uncalib2D_) {
      MEs_.erase(std::string("Uncalib2D"));
      MEs_.erase(std::string("Uncalib2DAmp"));
      MEs_.erase(std::string("Uncalib2DAmpError"));
      MEs_.erase(std::string("Uncalib2DPedestal"));
      MEs_.erase(std::string("Uncalib2DJitter"));
      MEs_.erase(std::string("Uncalib2DJitterError"));
      MEs_.erase(std::string("Uncalib2DChi2"));
      MEs_.erase(std::string("Uncalib2DOOTAmp"));
      MEs_.erase(std::string("Uncalib2DFlags"));
    }
    if (!rechit2D_) {
      MEs_.erase(std::string("RecHit2D"));
      MEs_.erase(std::string("RecHit2DEnergy"));
      MEs_.erase(std::string("RecHit2DTime"));
      MEs_.erase(std::string("RecHit2DFlags"));
    }
  }

  bool GpuTask::filterRunType(short const* runType) {
    for (unsigned iFED(0); iFED != ecaldqm::nDCC; iFED++) {
      if (runType[iFED] == EcalDCCHeaderBlock::COSMIC || runType[iFED] == EcalDCCHeaderBlock::MTCC ||
          runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL || runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
          runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL || runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL)
        return true;
    }

    return false;
  }

  void GpuTask::beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) {
    EBCpuDigis_ = nullptr;
    EECpuDigis_ = nullptr;
    EBCpuUncalibRecHits_ = nullptr;
    EECpuUncalibRecHits_ = nullptr;
    EBCpuRecHits_ = nullptr;
    EECpuRecHits_ = nullptr;
  }

  template <typename DigiCollection>
  void GpuTask::runOnCpuDigis(DigiCollection const& cpuDigis, Collections collection) {
    MESet& meDigiCpu(MEs_.at("DigiCpu"));
    MESet& meDigiCpuAmplitude(MEs_.at("DigiCpuAmplitude"));

    int iSubdet(collection == kEBCpuDigi ? EcalBarrel : EcalEndcap);

    // Save CpuDigis for comparison with GpuDigis
    // "if constexpr" ensures cpuDigis is the correct type at compile time
    if constexpr (std::is_same_v<DigiCollection, EBDigiCollection>) {
      assert(iSubdet == EcalBarrel);
      EBCpuDigis_ = &cpuDigis;
    } else {
      assert(iSubdet == EcalEndcap);
      EECpuDigis_ = &cpuDigis;
    }

    unsigned nCpuDigis(cpuDigis.size());
    meDigiCpu.fill(getEcalDQMSetupObjects(), iSubdet, nCpuDigis);

    for (auto const& cpuDigi : cpuDigis) {
      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame (digis) in the constructor
      EcalDataFrame cpuDataFrame(cpuDigi);

      for (unsigned iSample = 0; iSample < ecalPh1::sampleSize; iSample++) {
        static_cast<MESetMulti&>(meDigiCpuAmplitude).use(iSample);

        int cpuAmp(cpuDataFrame.sample(iSample).adc());
        meDigiCpuAmplitude.fill(getEcalDQMSetupObjects(), iSubdet, cpuAmp);
      }
    }
  }

  template <typename DigiCollection>
  void GpuTask::runOnGpuDigis(DigiCollection const& gpuDigis, Collections collection) {
    MESet& meDigiGpuCpu(MEs_.at("DigiGpuCpu"));
    MESet& meDigiGpuCpuAmplitude(MEs_.at("DigiGpuCpuAmplitude"));

    int iSubdet(collection == kEBGpuDigi ? EcalBarrel : EcalEndcap);

    // Get CpuDigis saved from GpuTask::runOnCpuDigis() for this event
    // "if constexpr" ensures cpuDigis is the correct type at compile time
    // Note: gpuDigis is a collection and cpuDigis is a pointer to a collection (for historical reasons)
    DigiCollection const* cpuDigis;
    if constexpr (std::is_same_v<DigiCollection, EBDigiCollection>) {
      assert(iSubdet == EcalBarrel);
      cpuDigis = EBCpuDigis_;
    } else {
      assert(iSubdet == EcalEndcap);
      cpuDigis = EECpuDigis_;
    }

    if (!cpuDigis) {
      edm::LogWarning("EcalDQM") << "GpuTask: Did not find " << ((iSubdet == EcalBarrel) ? "EB" : "EE")
                                 << "CpuDigis Collection. Aborting runOnGpuDigis\n";
      return;
    }

    unsigned nGpuDigis(gpuDigis.size());
    unsigned nCpuDigis(cpuDigis->size());

    meDigiGpuCpu.fill(getEcalDQMSetupObjects(), iSubdet, nGpuDigis - nCpuDigis);

    if (digi1D_) {
      MESet& meDigiGpu(MEs_.at("DigiGpu"));
      meDigiGpu.fill(getEcalDQMSetupObjects(), iSubdet, nGpuDigis);
    }

    if (digi2D_) {
      MESet& meDigi2D(MEs_.at("Digi2D"));
      meDigi2D.fill(getEcalDQMSetupObjects(), iSubdet, nCpuDigis, nGpuDigis);
    }

    for (auto const& gpuDigi : gpuDigis) {
      // Find CpuDigi with matching DetId
      DetId gpuId(gpuDigi.id());
      typename DigiCollection::const_iterator cpuItr(cpuDigis->find(gpuId));
      if (cpuItr == cpuDigis->end()) {
        edm::LogWarning("EcalDQM") << "GpuTask: Did not find CpuDigi DetId " << gpuId.rawId() << " in CPU collection\n";
        continue;
      }

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame (digis) in the constructor
      EcalDataFrame gpuDataFrame(gpuDigi);
      EcalDataFrame cpuDataFrame(*cpuItr);

      for (unsigned iSample = 0; iSample < ecalPh1::sampleSize; iSample++) {
        static_cast<MESetMulti&>(meDigiGpuCpuAmplitude).use(iSample);

        int gpuAmp(gpuDataFrame.sample(iSample).adc());
        int cpuAmp(cpuDataFrame.sample(iSample).adc());

        meDigiGpuCpuAmplitude.fill(getEcalDQMSetupObjects(), iSubdet, gpuAmp - cpuAmp);

        if (digi1D_) {
          MESet& meDigiGpuAmplitude(MEs_.at("DigiGpuAmplitude"));
          static_cast<MESetMulti&>(meDigiGpuAmplitude).use(iSample);
          meDigiGpuAmplitude.fill(getEcalDQMSetupObjects(), iSubdet, gpuAmp);
        }

        if (digi2D_) {
          MESet& meDigi2DAmplitude(MEs_.at("Digi2DAmplitude"));
          static_cast<MESetMulti&>(meDigi2DAmplitude).use(iSample);
          meDigi2DAmplitude.fill(getEcalDQMSetupObjects(), iSubdet, cpuAmp, gpuAmp);
        }
      }
    }
  }

  void GpuTask::runOnCpuUncalibRecHits(EcalUncalibratedRecHitCollection const& cpuHits, Collections collection) {
    MESet& meUncalibCpu(MEs_.at("UncalibCpu"));
    MESet& meUncalibCpuAmp(MEs_.at("UncalibCpuAmp"));
    MESet& meUncalibCpuAmpError(MEs_.at("UncalibCpuAmpError"));
    MESet& meUncalibCpuPedestal(MEs_.at("UncalibCpuPedestal"));
    MESet& meUncalibCpuJitter(MEs_.at("UncalibCpuJitter"));
    MESet& meUncalibCpuJitterError(MEs_.at("UncalibCpuJitterError"));
    MESet& meUncalibCpuChi2(MEs_.at("UncalibCpuChi2"));
    MESet& meUncalibCpuOOTAmp(MEs_.at("UncalibCpuOOTAmp"));
    MESet& meUncalibCpuFlags(MEs_.at("UncalibCpuFlags"));

    int iSubdet(collection == kEBCpuUncalibRecHit ? EcalBarrel : EcalEndcap);

    // Save CpuUncalibRecHits for comparison with GpuUncalibRecHits
    if (iSubdet == EcalBarrel)
      EBCpuUncalibRecHits_ = &cpuHits;
    else
      EECpuUncalibRecHits_ = &cpuHits;

    unsigned nCpuHits(cpuHits.size());
    meUncalibCpu.fill(getEcalDQMSetupObjects(), iSubdet, nCpuHits);

    for (auto const& cpuHit : cpuHits) {
      float cpuAmp(cpuHit.amplitude());
      float cpuAmpError(cpuHit.amplitudeError());
      float cpuPedestal(cpuHit.pedestal());
      float cpuJitter(cpuHit.jitter());
      float cpuJitterError(cpuHit.jitterError());
      float cpuChi2(cpuHit.chi2());
      uint32_t cpuFlags(cpuHit.flags());

      if (cpuJitterError == 10000)  // Set this so 10000 (special value) shows up in last bin
        cpuJitterError = 0.249999;

      meUncalibCpuAmp.fill(getEcalDQMSetupObjects(), iSubdet, cpuAmp);
      meUncalibCpuAmpError.fill(getEcalDQMSetupObjects(), iSubdet, cpuAmpError);
      meUncalibCpuPedestal.fill(getEcalDQMSetupObjects(), iSubdet, cpuPedestal);
      meUncalibCpuJitter.fill(getEcalDQMSetupObjects(), iSubdet, cpuJitter);
      meUncalibCpuJitterError.fill(getEcalDQMSetupObjects(), iSubdet, cpuJitterError);
      meUncalibCpuChi2.fill(getEcalDQMSetupObjects(), iSubdet, cpuChi2);
      meUncalibCpuFlags.fill(getEcalDQMSetupObjects(), iSubdet, cpuFlags);

      for (unsigned iAmp = 0; iAmp < uncalibOOTAmps_.size(); iAmp++) {
        static_cast<MESetMulti&>(meUncalibCpuOOTAmp).use(iAmp);

        // Get corresponding OOT Amplitude
        int cpuOOTAmp(cpuHit.outOfTimeAmplitude(uncalibOOTAmps_[iAmp]));
        meUncalibCpuOOTAmp.fill(getEcalDQMSetupObjects(), iSubdet, cpuOOTAmp);
      }
    }
  }

  void GpuTask::runOnGpuUncalibRecHits(EcalUncalibratedRecHitCollection const& gpuHits, Collections collection) {
    MESet& meUncalibGpuCpu(MEs_.at("UncalibGpuCpu"));
    MESet& meUncalibGpuCpuAmp(MEs_.at("UncalibGpuCpuAmp"));
    MESet& meUncalibGpuCpuAmpError(MEs_.at("UncalibGpuCpuAmpError"));
    MESet& meUncalibGpuCpuPedestal(MEs_.at("UncalibGpuCpuPedestal"));
    MESet& meUncalibGpuCpuJitter(MEs_.at("UncalibGpuCpuJitter"));
    MESet& meUncalibGpuCpuJitterError(MEs_.at("UncalibGpuCpuJitterError"));
    MESet& meUncalibGpuCpuChi2(MEs_.at("UncalibGpuCpuChi2"));
    MESet& meUncalibGpuCpuOOTAmp(MEs_.at("UncalibGpuCpuOOTAmp"));
    MESet& meUncalibGpuCpuFlags(MEs_.at("UncalibGpuCpuFlags"));

    int iSubdet(collection == kEBGpuUncalibRecHit ? EcalBarrel : EcalEndcap);

    // Get CpuUncalibRecHits saved from GpuTask::runOnCpuUncalibRecHits() for this event
    // Note: _gpuHits is a collection and cpuHits is a pointer to a collection (for historical reasons)
    EcalUncalibratedRecHitCollection const* cpuHits =
        (iSubdet == EcalBarrel) ? EBCpuUncalibRecHits_ : EECpuUncalibRecHits_;
    if (!cpuHits) {
      edm::LogWarning("EcalDQM") << "GpuTask: Did not find " << ((iSubdet == EcalBarrel) ? "EB" : "EE")
                                 << "CpuUncalibRecHits Collection. Aborting runOnGpuUncalibRecHits\n";
      return;
    }

    unsigned nGpuHits(gpuHits.size());
    unsigned nCpuHits(cpuHits->size());

    meUncalibGpuCpu.fill(getEcalDQMSetupObjects(), iSubdet, nGpuHits - nCpuHits);

    if (uncalib1D_) {
      MESet& meUncalibGpu(MEs_.at("UncalibGpu"));
      meUncalibGpu.fill(getEcalDQMSetupObjects(), iSubdet, nGpuHits);
    }

    if (uncalib2D_) {
      MESet& meUncalib2D(MEs_.at("Uncalib2D"));
      meUncalib2D.fill(getEcalDQMSetupObjects(), iSubdet, nCpuHits, nGpuHits);
    }

    for (auto const& gpuHit : gpuHits) {
      // Find CpuUncalibRecHit with matching DetId
      DetId gpuId(gpuHit.id());
      EcalUncalibratedRecHitCollection::const_iterator cpuItr(cpuHits->find(gpuId));
      if (cpuItr == cpuHits->end()) {
        edm::LogWarning("EcalDQM") << "GpuTask: Did not find GpuUncalibRecHit DetId " << gpuId.rawId()
                                   << " in CPU collection\n";
        continue;
      }

      float gpuAmp(gpuHit.amplitude());
      float gpuAmpError(gpuHit.amplitudeError());
      float gpuPedestal(gpuHit.pedestal());
      float gpuJitter(gpuHit.jitter());
      float gpuJitterError(gpuHit.jitterError());
      float gpuChi2(gpuHit.chi2());
      uint32_t gpuFlags(gpuHit.flags());

      if (gpuJitterError == 10000)  // Set this so 10000 (special value) shows up in last bin
        gpuJitterError = 0.249999;

      float cpuAmp(cpuItr->amplitude());
      float cpuAmpError(cpuItr->amplitudeError());
      float cpuPedestal(cpuItr->pedestal());
      float cpuJitter(cpuItr->jitter());
      float cpuJitterError(cpuItr->jitterError());
      float cpuChi2(cpuItr->chi2());
      uint32_t cpuFlags(cpuItr->flags());

      if (cpuJitterError == 10000)  // Set this so 10000 (special value) shows up in last bin
        cpuJitterError = 0.249999;

      meUncalibGpuCpuAmp.fill(getEcalDQMSetupObjects(), iSubdet, gpuAmp - cpuAmp);
      meUncalibGpuCpuAmpError.fill(getEcalDQMSetupObjects(), iSubdet, gpuAmpError - cpuAmpError);
      meUncalibGpuCpuPedestal.fill(getEcalDQMSetupObjects(), iSubdet, gpuPedestal - cpuPedestal);
      meUncalibGpuCpuJitter.fill(getEcalDQMSetupObjects(), iSubdet, gpuJitter - cpuJitter);
      meUncalibGpuCpuJitterError.fill(getEcalDQMSetupObjects(), iSubdet, gpuJitterError - cpuJitterError);
      meUncalibGpuCpuChi2.fill(getEcalDQMSetupObjects(), iSubdet, gpuChi2 - cpuChi2);
      meUncalibGpuCpuFlags.fill(getEcalDQMSetupObjects(), iSubdet, gpuFlags - cpuFlags);

      if (uncalib1D_) {
        MESet& meUncalibGpuAmp(MEs_.at("UncalibGpuAmp"));
        MESet& meUncalibGpuAmpError(MEs_.at("UncalibGpuAmpError"));
        MESet& meUncalibGpuPedestal(MEs_.at("UncalibGpuPedestal"));
        MESet& meUncalibGpuJitter(MEs_.at("UncalibGpuJitter"));
        MESet& meUncalibGpuJitterError(MEs_.at("UncalibGpuJitterError"));
        MESet& meUncalibGpuChi2(MEs_.at("UncalibGpuChi2"));
        MESet& meUncalibGpuFlags(MEs_.at("UncalibGpuFlags"));

        meUncalibGpuAmp.fill(getEcalDQMSetupObjects(), iSubdet, gpuAmp);
        meUncalibGpuAmpError.fill(getEcalDQMSetupObjects(), iSubdet, gpuAmpError);
        meUncalibGpuPedestal.fill(getEcalDQMSetupObjects(), iSubdet, gpuPedestal);
        meUncalibGpuJitter.fill(getEcalDQMSetupObjects(), iSubdet, gpuJitter);
        meUncalibGpuJitterError.fill(getEcalDQMSetupObjects(), iSubdet, gpuJitterError);
        meUncalibGpuChi2.fill(getEcalDQMSetupObjects(), iSubdet, gpuChi2);
        meUncalibGpuFlags.fill(getEcalDQMSetupObjects(), iSubdet, gpuFlags);
      }

      if (uncalib2D_) {
        MESet& meUncalib2DAmp(MEs_.at("Uncalib2DAmp"));
        MESet& meUncalib2DAmpError(MEs_.at("Uncalib2DAmpError"));
        MESet& meUncalib2DPedestal(MEs_.at("Uncalib2DPedestal"));
        MESet& meUncalib2DJitter(MEs_.at("Uncalib2DJitter"));
        MESet& meUncalib2DJitterError(MEs_.at("Uncalib2DJitterError"));
        MESet& meUncalib2DChi2(MEs_.at("Uncalib2DChi2"));
        MESet& meUncalib2DFlags(MEs_.at("Uncalib2DFlags"));

        meUncalib2DAmp.fill(getEcalDQMSetupObjects(), iSubdet, cpuAmp, gpuAmp);
        meUncalib2DAmpError.fill(getEcalDQMSetupObjects(), iSubdet, cpuAmpError, gpuAmpError);
        meUncalib2DPedestal.fill(getEcalDQMSetupObjects(), iSubdet, cpuPedestal, gpuPedestal);
        meUncalib2DJitter.fill(getEcalDQMSetupObjects(), iSubdet, cpuJitter, gpuJitter);
        meUncalib2DJitterError.fill(getEcalDQMSetupObjects(), iSubdet, cpuJitterError, gpuJitterError);
        meUncalib2DChi2.fill(getEcalDQMSetupObjects(), iSubdet, cpuChi2, gpuChi2);
        meUncalib2DFlags.fill(getEcalDQMSetupObjects(), iSubdet, cpuFlags, gpuFlags);
      }

      for (unsigned iAmp = 0; iAmp < uncalibOOTAmps_.size(); iAmp++) {
        static_cast<MESetMulti&>(meUncalibGpuCpuOOTAmp).use(iAmp);

        // Get corresponding OOT Amplitude
        int gpuOOTAmp(gpuHit.outOfTimeAmplitude(uncalibOOTAmps_[iAmp]));
        int cpuOOTAmp(cpuItr->outOfTimeAmplitude(uncalibOOTAmps_[iAmp]));

        meUncalibGpuCpuOOTAmp.fill(getEcalDQMSetupObjects(), iSubdet, gpuOOTAmp - cpuOOTAmp);

        if (uncalib1D_) {
          MESet& meUncalibGpuOOTAmp(MEs_.at("UncalibGpuOOTAmp"));
          static_cast<MESetMulti&>(meUncalibGpuOOTAmp).use(iAmp);
          meUncalibGpuOOTAmp.fill(getEcalDQMSetupObjects(), iSubdet, gpuOOTAmp);
        }

        if (uncalib2D_) {
          MESet& meUncalib2DOOTAmp(MEs_.at("Uncalib2DOOTAmp"));
          static_cast<MESetMulti&>(meUncalib2DOOTAmp).use(iAmp);
          meUncalib2DOOTAmp.fill(getEcalDQMSetupObjects(), iSubdet, cpuOOTAmp, gpuOOTAmp);
        }
      }
    }
  }

  void GpuTask::runOnCpuRecHits(EcalRecHitCollection const& cpuHits, Collections collection) {
    MESet& meRecHitCpu(MEs_.at("RecHitCpu"));
    MESet& meRecHitCpuEnergy(MEs_.at("RecHitCpuEnergy"));
    MESet& meRecHitCpuTime(MEs_.at("RecHitCpuTime"));
    MESet& meRecHitCpuFlags(MEs_.at("RecHitCpuFlags"));

    int iSubdet(collection == kEBCpuRecHit ? EcalBarrel : EcalEndcap);

    // Save CpuRecHits for comparison with GpuRecHits
    if (iSubdet == EcalBarrel)
      EBCpuRecHits_ = &cpuHits;
    else
      EECpuRecHits_ = &cpuHits;

    unsigned nCpuHits(cpuHits.size());
    meRecHitCpu.fill(getEcalDQMSetupObjects(), iSubdet, nCpuHits);

    for (auto const& cpuHit : cpuHits) {
      float cpuEnergy(cpuHit.energy());
      float cpuTime(cpuHit.time());
      uint32_t cpuFlags(cpuHit.flagsBits());

      meRecHitCpuEnergy.fill(getEcalDQMSetupObjects(), iSubdet, cpuEnergy);
      meRecHitCpuTime.fill(getEcalDQMSetupObjects(), iSubdet, cpuTime);
      meRecHitCpuFlags.fill(getEcalDQMSetupObjects(), iSubdet, cpuFlags);
    }
  }

  void GpuTask::runOnGpuRecHits(EcalRecHitCollection const& gpuHits, Collections collection) {
    MESet& meRecHitGpuCpu(MEs_.at("RecHitGpuCpu"));
    MESet& meRecHitGpuCpuEnergy(MEs_.at("RecHitGpuCpuEnergy"));
    MESet& meRecHitGpuCpuTime(MEs_.at("RecHitGpuCpuTime"));
    MESet& meRecHitGpuCpuFlags(MEs_.at("RecHitGpuCpuFlags"));

    int iSubdet(collection == kEBGpuRecHit ? EcalBarrel : EcalEndcap);

    // Get CpuRecHits saved from GpuTask::runOnCpuRecHits() for this event
    // Note: _gpuHits is a collection and cpuHits is a pointer to a collection (for historical reasons)
    EcalRecHitCollection const* cpuHits = (iSubdet == EcalBarrel) ? EBCpuRecHits_ : EECpuRecHits_;
    if (!cpuHits) {
      edm::LogWarning("EcalDQM") << "GpuTask: Did not find " << ((iSubdet == EcalBarrel) ? "EB" : "EE")
                                 << "CpuRecHits Collection. Aborting runOnGpuRecHits\n";
      return;
    }

    unsigned nGpuHits(gpuHits.size());
    unsigned nCpuHits(cpuHits->size());

    meRecHitGpuCpu.fill(getEcalDQMSetupObjects(), iSubdet, nGpuHits - nCpuHits);

    if (rechit1D_) {
      MESet& meRecHitGpu(MEs_.at("RecHitGpu"));
      meRecHitGpu.fill(getEcalDQMSetupObjects(), iSubdet, nGpuHits);
    }
    if (rechit2D_) {
      MESet& meRecHit2D(MEs_.at("RecHit2D"));
      meRecHit2D.fill(getEcalDQMSetupObjects(), iSubdet, nCpuHits, nGpuHits);
    }

    for (auto const& gpuHit : gpuHits) {
      // Find CpuRecHit with matching DetId
      DetId gpuId(gpuHit.detid());
      EcalRecHitCollection::const_iterator cpuItr(cpuHits->find(gpuId));
      if (cpuItr == cpuHits->end()) {
        edm::LogWarning("EcalDQM") << "GpuTask: Did not find GpuRecHit DetId " << gpuId.rawId()
                                   << " in CPU collection\n";
        continue;
      }

      float gpuEnergy(gpuHit.energy());
      float gpuTime(gpuHit.time());
      uint32_t gpuFlags(gpuHit.flagsBits());

      float cpuEnergy(cpuItr->energy());
      float cpuTime(cpuItr->time());
      uint32_t cpuFlags(cpuItr->flagsBits());

      meRecHitGpuCpuEnergy.fill(getEcalDQMSetupObjects(), iSubdet, gpuEnergy - cpuEnergy);
      meRecHitGpuCpuTime.fill(getEcalDQMSetupObjects(), iSubdet, gpuTime - cpuTime);
      meRecHitGpuCpuFlags.fill(getEcalDQMSetupObjects(), iSubdet, gpuFlags - cpuFlags);

      if (rechit1D_) {
        MESet& meRecHitGpuEnergy(MEs_.at("RecHitGpuEnergy"));
        MESet& meRecHitGpuTime(MEs_.at("RecHitGpuTime"));
        MESet& meRecHitGpuFlags(MEs_.at("RecHitGpuFlags"));

        meRecHitGpuEnergy.fill(getEcalDQMSetupObjects(), iSubdet, gpuEnergy);
        meRecHitGpuTime.fill(getEcalDQMSetupObjects(), iSubdet, gpuTime);
        meRecHitGpuFlags.fill(getEcalDQMSetupObjects(), iSubdet, gpuFlags);
      }

      if (rechit2D_) {
        MESet& meRecHit2DEnergy(MEs_.at("RecHit2DEnergy"));
        MESet& meRecHit2DTime(MEs_.at("RecHit2DTime"));
        MESet& meRecHit2DFlags(MEs_.at("RecHit2DFlags"));

        meRecHit2DEnergy.fill(getEcalDQMSetupObjects(), iSubdet, cpuEnergy, gpuEnergy);
        meRecHit2DTime.fill(getEcalDQMSetupObjects(), iSubdet, cpuTime, gpuTime);
        meRecHit2DFlags.fill(getEcalDQMSetupObjects(), iSubdet, cpuFlags, gpuFlags);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(GpuTask);
}  // namespace ecaldqm
