/*
 *   File: DataFormats/Scalers/src/L1TriggerRates.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/L1TriggerRates.h"

#include <iostream>

L1TriggerRates::L1TriggerRates():
  version_(0),
  triggerNumberRate_(0.0),
  eventNumberRate_(0.0),
  physicsL1AcceptsRate_(0.0),
  physicsL1AcceptsRawRate_(0.0),
  randomL1AcceptsRate_(0.0),
  calibrationL1AcceptsRate_(0.0),
  technicalTriggersRate_(0.0),
  orbitNumberRate_(0.0),
  numberResetsRate_(0.0),
  deadTimePercent_(0.0),
  deadTimeActivePercent_(0.0),
  deadTimeActiveCalibrationPercent_(0.0),
  deadTimeActivePrivatePercent_(0.0),
  deadTimeActivePartitionPercent_(0.0),
  deadTimeActiveThrottlePercent_(0.0),
  lostBunchCrossingsPercent_(0.0),
  lostTriggersPercent_(0.0),
  lostTriggersActivePercent_(0.0),
  triggersRate_(L1TriggerScalers::nL1Triggers),
  triggerNumberRunRate_(0.0),
  eventNumberRunRate_(0.0),
  physicsL1AcceptsRunRate_(0.0),
  physicsL1AcceptsRawRunRate_(0.0),
  randomL1AcceptsRunRate_(0.0),
  calibrationL1AcceptsRunRate_(0.0),
  technicalTriggersRunRate_(0.0),
  orbitNumberRunRate_(0.0),
  numberResetsRunRate_(0.0),
  deadTimeRunPercent_(0.0),
  deadTimeActiveRunPercent_(0.0),
  deadTimeActiveCalibrationRunPercent_(0.0),
  deadTimeActivePrivateRunPercent_(0.0),
  deadTimeActivePartitionRunPercent_(0.0),
  deadTimeActiveThrottleRunPercent_(0.0),
  lostBunchCrossingsRunPercent_(0.0),
  lostTriggersRunPercent_(0.0),
  lostTriggersActiveRunPercent_(0.0),
  triggersRunRate_(L1TriggerScalers::nL1Triggers)
{ 
  collectionTimeSummary_.tv_sec = 0;
  collectionTimeSummary_.tv_nsec = 0;
  collectionTimeDetails_.tv_sec = 0;
  collectionTimeDetails_.tv_nsec = 0;
}

L1TriggerRates::L1TriggerRates(const L1TriggerScalers s)
{ 
  L1TriggerRates();
  computeRunRates(s);
}

L1TriggerRates::L1TriggerRates(const L1TriggerScalers s1,
			       const L1TriggerScalers s2)
{  
  L1TriggerRates();

  const L1TriggerScalers *t1 = &s1;
  const L1TriggerScalers *t2 = &s2;

  // Choose the later sample to be t2
  if ( t1->orbitNumber() > t2->orbitNumber())
  {
    t1 = &s2;
    t2 = &s1;
  }

  computeRunRates(*t2);
  computeRates(*t1,*t2);
}

L1TriggerRates::~L1TriggerRates() { } 


void L1TriggerRates::computeRates(const L1TriggerScalers t1,
				  const L1TriggerScalers t2)
{
  double deltaOrbit = (double)t2.orbitNumber() - (double)t1.orbitNumber();
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

    int length1 = t1.triggers().size();
    int length2 = t2.triggers().size();
    int minLength;
    ( length1 >= length2 ) ? minLength = length2 : minLength=length1;
    std::vector<unsigned int> triggers1 = t1.triggers();
    std::vector<unsigned int> triggers2 = t2.triggers();
    for ( int i=0; i<minLength; i++)
    {
      double rate = ((double)triggers2[i] -
		     (double)triggers1[i] ) / deltaT_;
      triggersRate_.push_back(rate);
    }
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

    int length = t.triggers().size();
    for ( int i=0; i<length; i++)
    {
      double rate = ((double)t.triggers()[i]) / deltaTRun_;
      triggersRunRate_.push_back(rate);
    }
  }
}


/// Pretty-print operator for L1TriggerRates
std::ostream& operator<<(std::ostream& s, const L1TriggerRates& c) 
{
  s << "L1TriggerRates Version: " << c.version() 
    << " Differential Rates in Hz, DeltaT: " <<
    c.deltaT() << " sec" << std::endl;
  char line[128];

  sprintf(line,
	  " TriggerNumber:     %e  EventNumber:             %e",
	  c.triggerNumberRate(), c.eventNumberRate());
  s << line << std::endl;

  sprintf(line,
	  " PhysicsL1Accepts:  %e  PhysicsL1AcceptsRaw:     %e",
	  c.physicsL1AcceptsRate(), c.physicsL1AcceptsRawRate());
  s << line << std::endl;

  sprintf(line,
	  " RandomL1Accepts:   %e  CalibrationL1Accepts:    %e",
	  c.randomL1AcceptsRate(), c.calibrationL1AcceptsRate());
  s << line << std::endl;

  sprintf(line,
	  " TechnicalTriggers: %e  OrbitNumber:             %e",
	  c.technicalTriggersRate(), c.orbitNumberRate());
  s << line << std::endl;

  sprintf(line,
	  " NumberResets:      %e  DeadTime:                %3.3f%%",
	  c.numberResetsRate(), c.deadTimePercent());
  s << line << std::endl;

  sprintf(line,
	  " DeadTimeActive:        %3.3f%%    DeadTimeActiveCalibration:  %3.3f%%",
	  c.deadTimeActivePercent(), 
	  c.deadTimeActiveCalibrationPercent());
  s << line << std::endl;

  sprintf(line,
	  " LostTriggers:          %3.3f%%    DeadTimeActivePartition:    %3.3f%%",
	  c.lostTriggersPercent(), 
	  c.deadTimeActivePartitionPercent());
  s << line << std::endl;

  sprintf(line,
	  " LostTriggersActive:    %3.3f%%    DeadTimeActiveThrottle:     %3.3f%%",
	  c.lostTriggersActivePercent(),
	  c.deadTimeActiveThrottlePercent());
  s << line << std::endl;

  sprintf(line,
	  " LostBunchCrossings:    %3.3f%%    DeadTimeActivePrivate:      %3.3f%%",
	  c.lostBunchCrossingsPercent(), 
	  c.deadTimeActivePrivatePercent());
  s << line << std::endl;

  std::vector<double> triggersRate = c.triggersRate();
  int length = triggersRate.size() / 4;
  for ( int i=0; i<length; i++)
  {
    sprintf(line," %3.3d:%e    %3.3d:%e    %3.3d:%e    %3.3d:%e",
	    i,              triggersRate[i], 
	    (i+length),     triggersRate[i+length], 
	    (i+(length*2)), triggersRate[i+(length*2)], 
	    (i+(length*3)), triggersRate[i+(length*3)]);
    s << line << std::endl;
  }


  // Run Average rates

  s << "L1TriggerRates Version: " << c.version() 
    << " Run Average Rates in Hz, DeltaT: " <<
    c.deltaTRun() << " sec" << std::endl;

  sprintf(line,
	  " TriggerNumber:     %e  EventNumber:             %e",
	  c.triggerNumberRunRate(), c.eventNumberRunRate());
  s << line << std::endl;

  sprintf(line,
	  " PhysicsL1Accepts:  %e  PhysicsL1AcceptsRaw:     %e",
	  c.physicsL1AcceptsRunRate(), c.physicsL1AcceptsRawRunRate());
  s << line << std::endl;

  sprintf(line,
	  " RandomL1Accepts:   %e  CalibrationL1Accepts:    %e",
	  c.randomL1AcceptsRunRate(), c.calibrationL1AcceptsRunRate());
  s << line << std::endl;

  sprintf(line,
	  " TechnicalTriggers: %e  OrbitNumber:             %e",
	  c.technicalTriggersRunRate(), c.orbitNumberRunRate());
  s << line << std::endl;

  sprintf(line,
	  " NumberResets:      %e  DeadTime:                %3.3f%%",
	  c.numberResetsRunRate(), c.deadTimeRunPercent());
  s << line << std::endl;

  sprintf(line,
	  " DeadTimeActive:        %3.3f%%    DeadTimeActiveCalibration:  %3.3f%%",
	  c.deadTimeActiveRunPercent(), 
	  c.deadTimeActiveCalibrationRunPercent());
  s << line << std::endl;

  sprintf(line,
	  " LostTriggers:          %3.3f%%    DeadTimeActivePartition:    %3.3f%%",
	  c.lostTriggersRunPercent(), 
	  c.deadTimeActivePartitionRunPercent());
  s << line << std::endl;

  sprintf(line,
	  " LostTriggersActive:    %3.3f%%    DeadTimeActiveThrottle:     %3.3f%%",
	  c.lostTriggersActiveRunPercent(),
	  c.deadTimeActiveThrottleRunPercent());
  s << line << std::endl;

  sprintf(line,
	  " LostBunchCrossings:    %3.3f%%    DeadTimeActivePrivate:      %3.3f%%",
	  c.lostBunchCrossingsRunPercent(), 
	  c.deadTimeActivePrivateRunPercent());
  s << line << std::endl;

  std::vector<double> triggersRunRate = c.triggersRunRate();
  length = triggersRunRate.size() / 4;
  for ( int i=0; i<length; i++)
  {
    sprintf(line," %3.3d:%e    %3.3d:%e    %3.3d:%e    %3.3d:%e",
	    i,              triggersRunRate[i], 
	    (i+length),     triggersRunRate[i+length], 
	    (i+(length*2)), triggersRunRate[i+(length*2)], 
	    (i+(length*3)), triggersRunRate[i+(length*3)]);
    s << line << std::endl;
  }

  return s;
}
