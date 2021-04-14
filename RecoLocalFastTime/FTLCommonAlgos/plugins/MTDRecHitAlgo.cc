#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDRecHitAlgoBase.h"

#include "RecoLocalFastTime/Records/interface/MTDTimeCalibRecord.h"
#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDTimeCalib.h"

class MTDRecHitAlgo : public MTDRecHitAlgoBase {
public:
  /// Constructor
  MTDRecHitAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes);

  /// Destructor
  ~MTDRecHitAlgo() override {}

  /// get event and eventsetup information
  void getEvent(const edm::Event&) final {}
  void getEventSetup(const edm::EventSetup&) final;

  /// make the rec hit
  FTLRecHit makeRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags) const final;

private:
  double thresholdToKeep_, calibration_;
  const MTDTimeCalib* time_calib_;
  edm::ESGetToken<MTDTimeCalib, MTDTimeCalibRecord> tcToken_;
};

MTDRecHitAlgo::MTDRecHitAlgo(const edm::ParameterSet& conf, edm::ConsumesCollector& sumes)
    : MTDRecHitAlgoBase(conf, sumes),
      thresholdToKeep_(conf.getParameter<double>("thresholdToKeep")),
      calibration_(conf.getParameter<double>("calibrationConstant")) {
  tcToken_ = sumes.esConsumes<MTDTimeCalib, MTDTimeCalibRecord>(edm::ESInputTag("", "MTDTimeCalib"));
}

void MTDRecHitAlgo::getEventSetup(const edm::EventSetup& es) {
  auto pTC = es.getHandle(tcToken_);
  time_calib_ = pTC.product();
}

FTLRecHit MTDRecHitAlgo::makeRecHit(const FTLUncalibratedRecHit& uRecHit, uint32_t& flags) const {
  unsigned char flagsWord = uRecHit.flags();
  float timeError = uRecHit.timeError();

  float energy = 0.;
  float time = 0.;

  /// position and positionError in unit cm
  float position = -1.f;
  float positionError = -1.f;

  switch (flagsWord) {
    // BTL bar geometry with only the right SiPM information available
    case 0x2: {
      energy = uRecHit.amplitude().second;
      time = uRecHit.time().second;

      break;
    }
    // BTL bar geometry with left and right SiPMs information available
    case 0x3: {
      energy = 0.5 * (uRecHit.amplitude().first + uRecHit.amplitude().second);
      time = 0.5 * (uRecHit.time().first + uRecHit.time().second);

      position = uRecHit.position();
      positionError = uRecHit.positionError();

      break;
    }
    // ETL, BTL tile geometry, BTL bar geometry with only the left SiPM information available
    default: {
      energy = uRecHit.amplitude().first;
      time = uRecHit.time().first;

      break;
    }
  }

  // --- Energy calibration: for the time being this is just a conversion pC --> MeV
  energy *= calibration_;

  // --- Time calibration: for the time being just removes a time offset in BTL
  time += time_calib_->getTimeCalib(uRecHit.id());

  FTLRecHit rh(uRecHit.id(), uRecHit.row(), uRecHit.column(), energy, time, timeError, position, positionError);

  // Now fill flags
  // all rechits from the digitizer are "good" at present
  if (energy > thresholdToKeep_) {
    flags = FTLRecHit::kGood;
    rh.setFlag(flags);
  } else {
    flags = FTLRecHit::kKilled;
    rh.setFlag(flags);
  }

  return rh;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(MTDRecHitAlgoFactory, MTDRecHitAlgo, "MTDRecHitAlgo");
