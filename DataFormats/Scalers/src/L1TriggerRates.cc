
/*
 *   File: DataFormats/Scalers/src/L1TriggerRates.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/L1TriggerRates.h"

#include <iostream>

L1TriggerRates::L1TriggerRates():
  version_(0),
  triggerNumberRate_(0),
  eventNumberRate_(0),
  physicsL1AcceptsRate_(0),
  physicsL1AcceptsRawRate_(0),
  randomL1AcceptsRate_(0),
  calibrationL1AcceptsRate_(0),
  technicalTriggersRate_(0),
  orbitNumberRate_(0),
  numberResetsRate_(0),
  deadTimePercent_(0),
  deadTimeActivePercent_(0),
  deadTimeActiveCalibrationPercent_(0),
  deadTimeActivePrivatePercent_(0),
  deadTimeActivePartitionPercent_(0),
  deadTimeActiveThrottlePercent_(0),
  lostBunchCrossingsPercent_(0),
  lostTriggersPercent_(0),
  lostTriggersActivePercent_(0),
  triggersRate_(L1TriggerScalers::nL1Triggers),
  triggerNumberRunRate_(0),
  eventNumberRunRate_(0),
  physicsL1AcceptsRunRate_(0),
  physicsL1AcceptsRawRunRate_(0),
  randomL1AcceptsRunRate_(0),
  calibrationL1AcceptsRunRate_(0),
  technicalTriggersRunRate_(0),
  orbitNumberRunRate_(0),
  numberResetsRunRate_(0),
  deadTimeRunPercent_(0),
  deadTimeActiveRunPercent_(0),
  deadTimeActiveCalibrationRunPercent_(0),
  deadTimeActivePrivateRunPercent_(0),
  deadTimeActivePartitionRunPercent_(0),
  deadTimeActiveThrottleRunPercent_(0),
  lostBunchCrossingsRunPercent_(0),
  lostTriggersRunPercent_(0),
  lostTriggersActiveRunPercent_(0),
  triggersRunRate_(L1TriggerScalers::nL1Triggers)
{ 
  collectionTimeSummary_.tv_sec = 0;
  collectionTimeSummary_.tv_nsec = 0;
  collectionTimeDetails_.tv_sec = 0;
  collectionTimeDetails_.tv_nsec = 0;
}

L1TriggerRates::L1TriggerRates(const L1TriggerScalers s)
{ 
  computeRunRates(s);
}

void L1TriggerRates::computeRates(const L1TriggerScalers t1,
				  const L1TriggerScalers t2)
{
  double deltaOrbit = (double)t2.orbitNumber() - (double)t2.orbitNumber();
  if ( deltaOrbit > 0 )
  {
    // Convert orbits into crossings and time in seconds
    double deltaBC       = deltaOrbit * N_BX;
    double deltaBCActive = deltaOrbit * N_BX_ACTIVE;
    deltaT_              = deltaBC * BX_SPACING;
    deltaTActive_        = deltaBCActive * BX_SPACING;

    triggerNumberRate_ = ((double)t2.triggerNumber() 
			  - (double)t1.triggerNumber()) / deltaT_;
    eventNumberRate_ = ((double)t2.eventNumber() 
			- (double)t1.eventNumber()) / deltaT_;
    physicsL1AcceptsRate_ = ((double)t2.physicsL1Accepts() 
			     - (double)t1.physicsL1Accepts()) / deltaT_;
    physicsL1AcceptsRawRate_ = ((double)t2.physicsL1AcceptsRaw() 
	- (double)t1.physicsL1AcceptsRaw()) / deltaT_;
    randomL1AcceptsRate_ = ((double)t2.randomL1Accepts() 
			    - (double)t1.randomL1Accepts()) / deltaT_;
    calibrationL1AcceptsRate_ = ((double)t2.calibrationL1Accepts() 
				 - (double)t1.calibrationL1Accepts()) / deltaT_;
    technicalTriggersRate_ = ((double)t2.technicalTriggers() 
		     - (double)t1.technicalTriggers()) / deltaT_;
    orbitNumberRate_ = ((double)t2.orbitNumber() 
			- (double)t1.orbitNumber()) / deltaT_;
    numberResetsRate_ = ((double)t2.numberResets() 
			 - (double)t1.numberResets()) / deltaT_;
    
    deadTimePercent_ = 100.0 * ((double)t2.deadTime() 
				- (double)t1.deadTime()) / deltaBC;
    deadTimeActivePercent_ = 100.0 * ((double)t2.deadTimeActive() 
			   - (double)t1.deadTimeActive()) / deltaBCActive;
    deadTimeActiveCalibrationPercent_ = 100.0 * ((double)t2.deadTimeActiveCalibration() 
				      - (double)t1.deadTimeActiveCalibration()) / deltaBCActive;
    deadTimeActivePrivatePercent_ = 100.0 * ((double)t2.deadTimeActivePrivate() 
				  - (double)t1.deadTimeActivePrivate()) / deltaBCActive;
    deadTimeActivePartitionPercent_ = 100.0 * ((double)t2.deadTimeActivePartition() 
				    - (double)t1.deadTimeActivePartition()) / deltaBCActive;
    deadTimeActiveThrottlePercent_ = 100.0 * ((double)t2.deadTimeActiveThrottle() 
				   - (double)t1.deadTimeActiveThrottle()) / deltaBCActive;
    lostBunchCrossingsPercent_ = 100.0 * ((double)t2.lostBunchCrossings() 
			       - (double)t1.lostBunchCrossings()) / deltaBC;
    lostTriggersPercent_ = 100.0 * ((double)t2.lostTriggers() 
			 - (double)t1.lostTriggers()) / deltaBC;
    lostTriggersActivePercent_ = 100.0 * ((double)t2.lostTriggersActive() 
			       - (double)t1.lostTriggersActive()) / deltaBCActive;
  }
}

void L1TriggerRates::computeRunRates(const L1TriggerScalers t)
{
  version_ = t.version();

  collectionTimeSummary_.tv_sec  = t.collectionTimeSummary().tv_sec;
  collectionTimeSummary_.tv_nsec = t.collectionTimeSummary().tv_nsec;

  collectionTimeDetails_.tv_sec  = t.collectionTimeDetails().tv_sec;
  collectionTimeDetails_.tv_nsec = t.collectionTimeDetails().tv_nsec;

  double deltaOrbit = (double)t.orbitNumber();
  if ( deltaOrbit > 0 )
  {
    // Convert orbits into crossings and time in seconds
    double deltaBC       = deltaOrbit * N_BX;
    double deltaBCActive = deltaOrbit * N_BX_ACTIVE;
    deltaTRun_           = deltaBC * BX_SPACING;
    deltaTRunActive_     = deltaBCActive * BX_SPACING;

    triggerNumberRunRate_ = (double)t.triggerNumber() / deltaTRun_;
    eventNumberRunRate_ = (double)t.eventNumber() / deltaTRun_;
    physicsL1AcceptsRunRate_ = (double)t.physicsL1Accepts() / deltaTRun_;
    physicsL1AcceptsRawRunRate_ = (double)t.physicsL1AcceptsRaw() / deltaTRun_;
    randomL1AcceptsRunRate_ = (double)t.randomL1Accepts() / deltaTRun_;
    calibrationL1AcceptsRunRate_ = (double)t.calibrationL1Accepts() / deltaTRun_;
    technicalTriggersRunRate_ = (double)t.technicalTriggers() / deltaTRun_;
    orbitNumberRunRate_ = (double)t.orbitNumber() / deltaTRun_;
    numberResetsRunRate_ = (double)t.numberResets() / deltaTRun_;
    
    deadTimeRunPercent_ = 100.0 * (double)t.deadTime() / deltaBC;
    deadTimeActiveRunPercent_ = 100.0 * (double)t.deadTimeActive() / deltaBCActive;
    deadTimeActiveCalibrationRunPercent_ = 100.0 * (double)t.deadTimeActiveCalibration() / deltaBCActive;
    deadTimeActivePrivateRunPercent_ = 100.0 * (double)t.deadTimeActivePrivate() / deltaBCActive;
    deadTimeActivePartitionRunPercent_ = 100.0 * (double)t.deadTimeActivePartition() / deltaBCActive;
    deadTimeActiveThrottleRunPercent_ = 100.0 * (double)t.deadTimeActiveThrottle() / deltaBCActive;
    lostBunchCrossingsRunPercent_ = 100.0 * (double)t.lostBunchCrossings() / deltaBC;
    lostTriggersRunPercent_ = 100.0 * (double)t.lostTriggers() / deltaBC;
    lostTriggersActiveRunPercent_ = 100.0 * (double)t.lostTriggersActive() / deltaBCActive;
  }
}


L1TriggerRates::L1TriggerRates(const L1TriggerScalers s1,
			       const L1TriggerScalers s2)
{ 
  L1TriggerRates();

  L1TriggerScalers t1 = s1;
  L1TriggerScalers t2 = s2;

  // Choose the later sample to be t2
  if ( t1.orbitNumber() > t2.orbitNumber())
  {
    t1 = s2;
    t2 = s1;
  }

  computeRunRates(t2);
  computeRates(t1,t2);
}

L1TriggerRates::~L1TriggerRates() { } 


/// Pretty-print operator for L1TriggerRates
std::ostream& operator<<(std::ostream& s, const L1TriggerRates& c) 
{
  s << " L1TriggerRates";
  return s;
}
