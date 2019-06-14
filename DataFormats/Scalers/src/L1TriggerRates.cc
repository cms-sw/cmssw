/*
 *   File: DataFormats/Scalers/src/L1TriggerRates.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/L1TriggerRates.h"
#include "DataFormats/Scalers/interface/L1TriggerScalers.h"

#include <iostream>
#include <cstdio>

L1TriggerRates::L1TriggerRates()
    : version_(0),
      collectionTimeSummary_(0, 0),
      deltaT_(0),
      deltaTActive_(0),
      triggerNumberRate_(0.0),
      eventNumberRate_(0.0),
      finalTriggersDistributedRate_(0.0),
      finalTriggersGeneratedRate_(0.0),
      randomTriggersRate_(0.0),
      calibrationTriggersRate_(0.0),
      totalTestTriggersRate_(0.0),
      orbitNumberRate_(0.0),
      numberResetsRate_(0.0),
      deadTimePercent_(0.0),
      deadTimeActivePercent_(0.0),
      deadTimeActiveCalibrationPercent_(0.0),
      deadTimeActivePrivatePercent_(0.0),
      deadTimeActivePartitionPercent_(0.0),
      deadTimeActiveThrottlePercent_(0.0),
      deadTimeActiveTimeSlotPercent_(0.0),
      finalTriggersInvalidBCPercent_(0.0),
      lostFinalTriggersPercent_(0.0),
      lostFinalTriggersActivePercent_(0.0),
      triggersRate_(L1TriggerScalers::nL1Triggers),
      testTriggersRate_(L1TriggerScalers::nL1TestTriggers),
      triggerNumberRunRate_(0.0),
      eventNumberRunRate_(0.0),
      finalTriggersDistributedRunRate_(0.0),
      finalTriggersGeneratedRunRate_(0.0),
      randomTriggersRunRate_(0.0),
      calibrationTriggersRunRate_(0.0),
      totalTestTriggersRunRate_(0.0),
      orbitNumberRunRate_(0.0),
      numberResetsRunRate_(0.0),
      deadTimeRunPercent_(0.0),
      deadTimeActiveRunPercent_(0.0),
      deadTimeActiveCalibrationRunPercent_(0.0),
      deadTimeActivePrivateRunPercent_(0.0),
      deadTimeActivePartitionRunPercent_(0.0),
      deadTimeActiveThrottleRunPercent_(0.0),
      deadTimeActiveTimeSlotRunPercent_(0.0),
      finalTriggersInvalidBCRunPercent_(0.0),
      lostFinalTriggersRunPercent_(0.0),
      lostFinalTriggersActiveRunPercent_(0.0),
      collectionTimeDetails_(0, 0),
      triggersRunRate_(L1TriggerScalers::nL1Triggers),
      testTriggersRunRate_(L1TriggerScalers::nL1TestTriggers) {}

L1TriggerRates::L1TriggerRates(L1TriggerScalers const& s) {
  L1TriggerRates();
  computeRunRates(s);
}

L1TriggerRates::L1TriggerRates(L1TriggerScalers const& s1, L1TriggerScalers const& s2) {
  L1TriggerRates();

  const L1TriggerScalers* t1 = &s1;
  const L1TriggerScalers* t2 = &s2;

  // Choose the later sample to be t2
  if (t1->orbitNumber() > t2->orbitNumber()) {
    t1 = &s2;
    t2 = &s1;
  }

  computeRunRates(*t2);
  computeRates(*t1, *t2);
}

L1TriggerRates::~L1TriggerRates() {}

void L1TriggerRates::computeRates(L1TriggerScalers const& t1, L1TriggerScalers const& t2) {
  double deltaOrbit = (double)t2.orbitNumber() - (double)t1.orbitNumber();
  if (deltaOrbit > 0) {
    // Convert orbits into crossings and time in seconds
    double deltaBC = deltaOrbit * N_BX;
    double deltaBCActive = deltaOrbit * N_BX_ACTIVE;
    deltaT_ = deltaBC * BX_SPACING;
    deltaTActive_ = deltaBCActive * BX_SPACING;

    triggerNumberRate_ = ((double)t2.triggerNumber() - (double)t1.triggerNumber()) / deltaT_;
    eventNumberRate_ = ((double)t2.eventNumber() - (double)t1.eventNumber()) / deltaT_;
    finalTriggersDistributedRate_ =
        ((double)t2.finalTriggersDistributed() - (double)t1.finalTriggersDistributed()) / deltaT_;
    finalTriggersGeneratedRate_ = ((double)t2.finalTriggersGenerated() - (double)t1.finalTriggersGenerated()) / deltaT_;
    randomTriggersRate_ = ((double)t2.randomTriggers() - (double)t1.randomTriggers()) / deltaT_;
    calibrationTriggersRate_ = ((double)t2.calibrationTriggers() - (double)t1.calibrationTriggers()) / deltaT_;
    totalTestTriggersRate_ = ((double)t2.totalTestTriggers() - (double)t1.totalTestTriggers()) / deltaT_;
    orbitNumberRate_ = ((double)t2.orbitNumber() - (double)t1.orbitNumber()) / deltaT_;
    numberResetsRate_ = ((double)t2.numberResets() - (double)t1.numberResets()) / deltaT_;

    deadTimePercent_ = 100.0 * ((double)t2.deadTime() - (double)t1.deadTime()) / deltaBC;
    deadTimeActivePercent_ = 100.0 * ((double)t2.deadTimeActive() - (double)t1.deadTimeActive()) / deltaBCActive;
    deadTimeActiveCalibrationPercent_ =
        100.0 * ((double)t2.deadTimeActiveCalibration() - (double)t1.deadTimeActiveCalibration()) / deltaBCActive;
    deadTimeActivePrivatePercent_ =
        100.0 * ((double)t2.deadTimeActivePrivate() - (double)t1.deadTimeActivePrivate()) / deltaBCActive;
    deadTimeActivePartitionPercent_ =
        100.0 * ((double)t2.deadTimeActivePartition() - (double)t1.deadTimeActivePartition()) / deltaBCActive;
    deadTimeActiveThrottlePercent_ =
        100.0 * ((double)t2.deadTimeActiveThrottle() - (double)t1.deadTimeActiveThrottle()) / deltaBCActive;
    deadTimeActiveTimeSlotPercent_ =
        100.0 * ((double)t2.deadTimeActiveTimeSlot() - (double)t1.deadTimeActiveTimeSlot()) / deltaBCActive;
    finalTriggersInvalidBCPercent_ =
        100.0 * ((double)t2.finalTriggersInvalidBC() - (double)t1.finalTriggersInvalidBC()) / deltaBC;
    lostFinalTriggersPercent_ = 100.0 * ((double)t2.lostFinalTriggers() - (double)t1.lostFinalTriggers()) / deltaBC;
    lostFinalTriggersActivePercent_ =
        100.0 * ((double)t2.lostFinalTriggersActive() - (double)t1.lostFinalTriggersActive()) / deltaBCActive;

    int length1 = t1.triggers().size();
    int length2 = t2.triggers().size();
    int minLength;
    (length1 >= length2) ? minLength = length2 : minLength = length1;
    std::vector<unsigned int> triggers1 = t1.triggers();
    std::vector<unsigned int> triggers2 = t2.triggers();
    for (int i = 0; i < minLength; i++) {
      double rate = ((double)triggers2[i] - (double)triggers1[i]) / deltaT_;
      triggersRate_.push_back(rate);
    }

    length1 = t1.testTriggers().size();
    length2 = t2.testTriggers().size();
    (length1 >= length2) ? minLength = length2 : minLength = length1;
    std::vector<unsigned int> testTriggers1 = t1.testTriggers();
    std::vector<unsigned int> testTriggers2 = t2.testTriggers();
    for (int i = 0; i < minLength; i++) {
      double rate = ((double)testTriggers2[i] - (double)testTriggers1[i]) / deltaT_;
      testTriggersRate_.push_back(rate);
    }
  }
}

void L1TriggerRates::computeRunRates(L1TriggerScalers const& t) {
  version_ = t.version();

  collectionTimeSummary_.set_tv_sec(static_cast<long>(t.collectionTimeSummary().tv_sec));
  collectionTimeSummary_.set_tv_nsec(t.collectionTimeSummary().tv_nsec);

  collectionTimeDetails_.set_tv_sec(static_cast<long>(t.collectionTimeDetails().tv_sec));
  collectionTimeDetails_.set_tv_nsec(t.collectionTimeDetails().tv_nsec);

  double deltaOrbit = (double)t.orbitNumber();
  if (deltaOrbit > 0) {
    // Convert orbits into crossings and time in seconds
    double deltaBC = deltaOrbit * N_BX;
    double deltaBCActive = deltaOrbit * N_BX_ACTIVE;
    deltaTRun_ = deltaBC * BX_SPACING;
    deltaTRunActive_ = deltaBCActive * BX_SPACING;

    triggerNumberRunRate_ = (double)t.triggerNumber() / deltaTRun_;
    eventNumberRunRate_ = (double)t.eventNumber() / deltaTRun_;
    finalTriggersDistributedRunRate_ = (double)t.finalTriggersDistributed() / deltaTRun_;
    finalTriggersGeneratedRunRate_ = (double)t.finalTriggersGenerated() / deltaTRun_;
    randomTriggersRunRate_ = (double)t.randomTriggers() / deltaTRun_;
    calibrationTriggersRunRate_ = (double)t.calibrationTriggers() / deltaTRun_;
    totalTestTriggersRunRate_ = (double)t.totalTestTriggers() / deltaTRun_;
    orbitNumberRunRate_ = (double)t.orbitNumber() / deltaTRun_;
    numberResetsRunRate_ = (double)t.numberResets() / deltaTRun_;

    deadTimeRunPercent_ = 100.0 * (double)t.deadTime() / deltaBC;
    deadTimeActiveRunPercent_ = 100.0 * (double)t.deadTimeActive() / deltaBCActive;
    deadTimeActiveCalibrationRunPercent_ = 100.0 * (double)t.deadTimeActiveCalibration() / deltaBCActive;
    deadTimeActivePrivateRunPercent_ = 100.0 * (double)t.deadTimeActivePrivate() / deltaBCActive;
    deadTimeActivePartitionRunPercent_ = 100.0 * (double)t.deadTimeActivePartition() / deltaBCActive;
    deadTimeActiveThrottleRunPercent_ = 100.0 * (double)t.deadTimeActiveThrottle() / deltaBCActive;
    deadTimeActiveTimeSlotRunPercent_ = 100.0 * (double)t.deadTimeActiveTimeSlot() / deltaBCActive;
    finalTriggersInvalidBCRunPercent_ = 100.0 * (double)t.finalTriggersInvalidBC() / deltaBC;
    lostFinalTriggersRunPercent_ = 100.0 * (double)t.lostFinalTriggers() / deltaBC;
    lostFinalTriggersActiveRunPercent_ = 100.0 * (double)t.lostFinalTriggersActive() / deltaBCActive;

    int length = t.triggers().size();
    for (int i = 0; i < length; i++) {
      double rate = ((double)t.triggers()[i]) / deltaTRun_;
      triggersRunRate_.push_back(rate);
    }
  }
}

/// Pretty-print operator for L1TriggerRates
std::ostream& operator<<(std::ostream& s, const L1TriggerRates& c) {
  s << "L1TriggerRates Version: " << c.version() << " Differential Rates in Hz, DeltaT: " << c.deltaT() << " sec"
    << std::endl;
  char line[128];

  sprintf(line, " TriggerNumber:       %e  EventNumber:             %e", c.triggerNumberRate(), c.eventNumberRate());
  s << line << std::endl;

  sprintf(line,
          " TriggersDistributed: %e  TriggersGenerated:     %e",
          c.finalTriggersDistributedRate(),
          c.finalTriggersGeneratedRate());
  s << line << std::endl;

  sprintf(line,
          " RandomTriggers:      %e  CalibrationTriggers:    %e",
          c.randomTriggersRate(),
          c.calibrationTriggersRate());
  s << line << std::endl;

  sprintf(
      line, " TotalTestTriggers:   %e  OrbitNumber:             %e", c.totalTestTriggersRate(), c.orbitNumberRate());
  s << line << std::endl;

  sprintf(
      line, " NumberResets:        %e  DeadTime:                %3.3f%%", c.numberResetsRate(), c.deadTimePercent());
  s << line << std::endl;

  sprintf(line,
          " DeadTimeActive:        %3.3f%%    DeadTimeActiveCalibration:  %3.3f%%",
          c.deadTimeActivePercent(),
          c.deadTimeActiveCalibrationPercent());
  s << line << std::endl;

  sprintf(line,
          " LostTriggers:          %3.3f%%    DeadTimeActivePartition:    %3.3f%%",
          c.lostFinalTriggersPercent(),
          c.deadTimeActivePartitionPercent());
  s << line << std::endl;

  sprintf(line,
          " LostTriggersActive:    %3.3f%%    DeadTimeActiveThrottle:     %3.3f%%",
          c.lostFinalTriggersActivePercent(),
          c.deadTimeActiveThrottlePercent());
  s << line << std::endl;

  sprintf(line,
          " TriggersInvalidBC:     %3.3f%%    DeadTimeActivePrivate:      %3.3f%%",
          c.finalTriggersInvalidBCPercent(),
          c.deadTimeActivePrivatePercent());
  s << line << std::endl;

  sprintf(line,
          "                                   DeadTimeActiveTimeSlot:     %3.3f%%",
          c.deadTimeActiveTimeSlotPercent());
  s << line << std::endl;

  std::vector<double> triggersRate = c.triggersRate();
  int length = triggersRate.size() / 4;
  for (int i = 0; i < length; i++) {
    sprintf(line,
            " %3.3d:%e    %3.3d:%e    %3.3d:%e    %3.3d:%e",
            i,
            triggersRate[i],
            (i + length),
            triggersRate[i + length],
            (i + (length * 2)),
            triggersRate[i + (length * 2)],
            (i + (length * 3)),
            triggersRate[i + (length * 3)]);
    s << line << std::endl;
  }

  std::vector<double> testTriggersRate = c.testTriggersRate();
  length = testTriggersRate.size() / 4;
  for (int i = 0; i < length; i++) {
    sprintf(line,
            " %3.3d:%e    %3.3d:%e    %3.3d:%e    %3.3d:%e",
            i,
            testTriggersRate[i],
            (i + length),
            testTriggersRate[i + length],
            (i + (length * 2)),
            testTriggersRate[i + (length * 2)],
            (i + (length * 3)),
            testTriggersRate[i + (length * 3)]);
    s << line << std::endl;
  }

  // Run Average rates

  s << "L1TriggerRates Version: " << c.version() << " Run Average Rates in Hz, DeltaT: " << c.deltaTRun() << " sec"
    << std::endl;

  sprintf(
      line, " TriggerNumber:     %e  EventNumber:             %e", c.triggerNumberRunRate(), c.eventNumberRunRate());
  s << line << std::endl;

  sprintf(line,
          " TriggersDistributed:  %e  TriggersGenerated:     %e",
          c.finalTriggersDistributedRunRate(),
          c.finalTriggersGeneratedRunRate());
  s << line << std::endl;

  sprintf(line,
          " RandomTriggers:   %e  CalibrationTriggers:    %e",
          c.randomTriggersRunRate(),
          c.calibrationTriggersRunRate());
  s << line << std::endl;

  sprintf(line,
          " TotalTestTriggers: %e  OrbitNumber:             %e",
          c.totalTestTriggersRunRate(),
          c.orbitNumberRunRate());
  s << line << std::endl;

  sprintf(line,
          " NumberResets:      %e  DeadTime:                %3.3f%%",
          c.numberResetsRunRate(),
          c.deadTimeRunPercent());
  s << line << std::endl;

  sprintf(line,
          " DeadTimeActive:        %3.3f%%    DeadTimeActiveCalibration:  %3.3f%%",
          c.deadTimeActiveRunPercent(),
          c.deadTimeActiveCalibrationRunPercent());
  s << line << std::endl;

  sprintf(line,
          " LostTriggers:          %3.3f%%    DeadTimeActivePartition:    %3.3f%%",
          c.lostFinalTriggersRunPercent(),
          c.deadTimeActivePartitionRunPercent());
  s << line << std::endl;

  sprintf(line,
          " LostTriggersActive:    %3.3f%%    DeadTimeActiveThrottle:     %3.3f%%",
          c.lostFinalTriggersActiveRunPercent(),
          c.deadTimeActiveThrottleRunPercent());
  s << line << std::endl;

  sprintf(line,
          " FinalTriggersInvalidBC:    %3.3f%%    DeadTimeActivePrivate:      %3.3f%%",
          c.finalTriggersInvalidBCRunPercent(),
          c.deadTimeActivePrivateRunPercent());
  s << line << std::endl;

  sprintf(line, " DeadTimeActiveTimeSlot:      %3.3f%%", c.deadTimeActiveTimeSlotRunPercent());
  s << line << std::endl;

  std::vector<double> triggersRunRate = c.triggersRunRate();
  length = triggersRunRate.size() / 4;
  for (int i = 0; i < length; i++) {
    sprintf(line,
            " %3.3d:%e    %3.3d:%e    %3.3d:%e    %3.3d:%e",
            i,
            triggersRunRate[i],
            (i + length),
            triggersRunRate[i + length],
            (i + (length * 2)),
            triggersRunRate[i + (length * 2)],
            (i + (length * 3)),
            triggersRunRate[i + (length * 3)]);
    s << line << std::endl;
  }

  return s;
}
