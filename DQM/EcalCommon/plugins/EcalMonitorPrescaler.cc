#include "DQM/EcalCommon/interface/EcalMonitorPrescaler.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <iostream>

const uint32_t EcalMonitorPrescaler::filterBits_[ecaldqm::nPrescalers] = {
    (1 << EcalDCCHeaderBlock::MTCC) | (1 << EcalDCCHeaderBlock::PHYSICS_GLOBAL) |
        (1 << EcalDCCHeaderBlock::PHYSICS_LOCAL),  // kPhysics
    (1 << EcalDCCHeaderBlock::COSMIC) | (1 << EcalDCCHeaderBlock::COSMICS_GLOBAL) |
        (1 << EcalDCCHeaderBlock::COSMICS_LOCAL),  // kCosmics
    (1 << EcalDCCHeaderBlock::LASER_STD) | (1 << EcalDCCHeaderBlock::LASER_GAP) | (1 << EcalDCCHeaderBlock::LED_STD) |
        (1 << EcalDCCHeaderBlock::LED_GAP) | (1 << EcalDCCHeaderBlock::PEDESTAL_STD) |
        (1 << EcalDCCHeaderBlock::PEDESTAL_GAP) | (1 << EcalDCCHeaderBlock::TESTPULSE_MGPA) |
        (1 << EcalDCCHeaderBlock::TESTPULSE_GAP) | (1 << EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN),  // kCalibration
    (1 << EcalDCCHeaderBlock::LASER_STD) | (1 << EcalDCCHeaderBlock::LASER_GAP),                     // kLaser
    (1 << EcalDCCHeaderBlock::LED_STD) | (1 << EcalDCCHeaderBlock::LED_GAP),                         // kLed
    (1 << EcalDCCHeaderBlock::TESTPULSE_MGPA) | (1 << EcalDCCHeaderBlock::TESTPULSE_GAP),            // kTestPulse
    (1 << EcalDCCHeaderBlock::PEDESTAL_STD) | (1 << EcalDCCHeaderBlock::PEDESTAL_GAP) |
        (1 << EcalDCCHeaderBlock::PEDESTAL_OFFSET_SCAN)  // kPedestal
};

EcalMonitorPrescaler::EcalMonitorPrescaler(edm::ParameterSet const &_ps)
    : EcalRawDataCollection_(
          consumes<EcalRawDataCollection>(_ps.getParameter<edm::InputTag>("EcalRawDataCollection"))) {
  prescalers_[ecaldqm::kPhysics] = _ps.getUntrackedParameter<unsigned>("physics", -1);
  prescalers_[ecaldqm::kCosmics] = _ps.getUntrackedParameter<unsigned>("cosmics", -1);
  prescalers_[ecaldqm::kCalibration] = _ps.getUntrackedParameter<unsigned>("calibration", -1);
  prescalers_[ecaldqm::kLaser] = _ps.getUntrackedParameter<unsigned>("laser", -1);
  prescalers_[ecaldqm::kLed] = _ps.getUntrackedParameter<unsigned>("led", -1);
  prescalers_[ecaldqm::kTestPulse] = _ps.getUntrackedParameter<unsigned>("testPulse", -1);
  prescalers_[ecaldqm::kPedestal] = _ps.getUntrackedParameter<unsigned>("pedestal", -1);

  // Backward compatibility
  prescalers_[ecaldqm::kPhysics] = std::min(
      prescalers_[ecaldqm::kPhysics], (unsigned int)(_ps.getUntrackedParameter<int>("occupancyPrescaleFactor", -1)));
  prescalers_[ecaldqm::kPhysics] = std::min(
      prescalers_[ecaldqm::kPhysics], (unsigned int)(_ps.getUntrackedParameter<int>("integrityPrescaleFactor", -1)));
  prescalers_[ecaldqm::kCosmics] = std::min(prescalers_[ecaldqm::kCosmics],
                                            (unsigned int)(_ps.getUntrackedParameter<int>("cosmicPrescaleFactor", -1)));
  prescalers_[ecaldqm::kLaser] =
      std::min(prescalers_[ecaldqm::kLaser], (unsigned int)(_ps.getUntrackedParameter<int>("laserPrescaleFactor", -1)));
  prescalers_[ecaldqm::kLed] =
      std::min(prescalers_[ecaldqm::kLed], (unsigned int)(_ps.getUntrackedParameter<int>("ledPrescaleFactor", -1)));
  prescalers_[ecaldqm::kPedestal] = std::min(
      prescalers_[ecaldqm::kPedestal], (unsigned int)(_ps.getUntrackedParameter<int>("pedestalPrescaleFactor", -1)));
  prescalers_[ecaldqm::kPedestal] =
      std::min(prescalers_[ecaldqm::kPedestal],
               (unsigned int)(_ps.getUntrackedParameter<int>("pedestalonlinePrescaleFactor", -1)));
  prescalers_[ecaldqm::kTestPulse] = std::min(
      prescalers_[ecaldqm::kTestPulse], (unsigned int)(_ps.getUntrackedParameter<int>("testpulsePrescaleFactor", -1)));
  prescalers_[ecaldqm::kPedestal] =
      std::min(prescalers_[ecaldqm::kPedestal],
               (unsigned int)(_ps.getUntrackedParameter<int>("pedestaloffsetPrescaleFactor", -1)));
  prescalers_[ecaldqm::kPhysics] = std::min(
      prescalers_[ecaldqm::kPhysics], (unsigned int)(_ps.getUntrackedParameter<int>("triggertowerPrescaleFactor", -1)));
  prescalers_[ecaldqm::kPhysics] = std::min(prescalers_[ecaldqm::kPhysics],
                                            (unsigned int)(_ps.getUntrackedParameter<int>("timingPrescaleFactor", -1)));
  prescalers_[ecaldqm::kPhysics] = std::min(
      prescalers_[ecaldqm::kPhysics], (unsigned int)(_ps.getUntrackedParameter<int>("physicsPrescaleFactor", -1)));
  prescalers_[ecaldqm::kPhysics] = std::min(
      prescalers_[ecaldqm::kPhysics], (unsigned int)(_ps.getUntrackedParameter<int>("clusterPrescaleFactor", -1)));
}

EcalMonitorPrescaler::~EcalMonitorPrescaler() {}

std::shared_ptr<ecaldqm::PrescaleCounter> EcalMonitorPrescaler::globalBeginRun(edm::Run const &,
                                                                               edm::EventSetup const &) const {
  return std::make_shared<ecaldqm::PrescaleCounter>();
}

bool EcalMonitorPrescaler::filter(edm::StreamID, edm::Event &_event, edm::EventSetup const &) const {
  edm::Handle<EcalRawDataCollection> dcchs;

  if (!_event.getByToken(EcalRawDataCollection_, dcchs)) {
    edm::LogWarning("EcalMonitorPrescaler") << "EcalRawDataCollection not available";
    return false;
  }

  uint32_t eventBits(0);
  for (EcalRawDataCollection::const_iterator dcchItr(dcchs->begin()); dcchItr != dcchs->end(); ++dcchItr)
    eventBits |= (1 << dcchItr->getRunType());

  for (unsigned iP(0); iP != ecaldqm::nPrescalers; ++iP) {
    if ((eventBits & filterBits_[iP]) != 0 &&
        ++(runCache(_event.getRun().index())->counters_[iP]) % prescalers_[iP] == 0)
      return true;
  }

  return false;
}

void EcalMonitorPrescaler::globalEndRun(edm::Run const &, edm::EventSetup const &) const {}

DEFINE_FWK_MODULE(EcalMonitorPrescaler);
