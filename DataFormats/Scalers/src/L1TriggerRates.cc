
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
  techTrigRate_(0),
  orbitNumberRate_(0),
  numberResetsRate_(0),
  deadTimeRate_(0),
  deadTimeActiveRate_(0),
  deadTimeActiveCalibrationRate_(0),
  deadTimeActivePrivateRate_(0),
  deadTimeActivePartitionRate_(0),
  deadTimeActiveThrottleRate_(0),
  lostBunchCrossingsRate_(0),
  lostTriggersRate_(0),
  lostTriggersActiveRate_(0),
  triggersRate_(L1TriggerScalers::nL1Triggers),
  triggerNumberRunRate_(0),
  eventNumberRunRate_(0),
  physicsL1AcceptsRunRate_(0),
  physicsL1AcceptsRawRunRate_(0),
  randomL1AcceptsRunRate_(0),
  calibrationL1AcceptsRunRate_(0),
  techTrigRunRate_(0),
  orbitNumberRunRate_(0),
  numberResetsRunRate_(0),
  deadTimeRunRate_(0),
  deadTimeActiveRunRate_(0),
  deadTimeActiveCalibrationRunRate_(0),
  deadTimeActivePrivateRunRate_(0),
  deadTimeActivePartitionRunRate_(0),
  deadTimeActiveThrottleRunRate_(0),
  lostBunchCrossingsRunRate_(0),
  lostTriggersRunRate_(0),
  lostTriggersActiveRunRate_(0),
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

}

void L1TriggerRates::computeRunRates(const L1TriggerScalers t)
{
  version_ = t.version();

//   collectionTimeSummary_.tv_sec = t.collectionTimeSummary().tv_sec;
//   collectionTimeSummary_.tv_nsec = ts.tv_nsec;

//   const struct timespec td = t.collectionTimeDetails();
//   collectionTimeDetails_.tv_sec = ts.tv_sec;
//   collectionTimeDetails_.tv_nsec = ts.tv_nsec;

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

//   triggerNumberRate             = ;
//   eventNumberRate               = ;
//   physicsL1AcceptsRate          = ;
//   physicsL1AcceptsRawRate       = ;
//   randomL1AcceptsRate           = ;
//   calibrationL1AcceptsRate      = ;
//   techTrigRate                  = ;
//   orbitNumberRate               = ;
//   numberResetsRate              = ;
//   deadTimeRate                  = ;
//   deadTimeActiveRate            = ;
//   deadTimeActiveCalibrationRate = ;
//   deadTimeActivePrivateRate     = ;
//   deadTimeActivePartitionRate   = ;
//   deadTimeActiveThrottleRate    = ;
//   lostBunchCrossingsRate        = ;
//   lostTriggersRate              = ;
//   lostTriggersActiveRate        = ;
  
}

L1TriggerRates::~L1TriggerRates() { } 


/// Pretty-print operator for L1TriggerRates
std::ostream& operator<<(std::ostream& s, const L1TriggerRates& c) 
{
  s << " L1TriggerRates: ";
  return s;
}
