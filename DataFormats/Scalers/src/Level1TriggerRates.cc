/*
 *   File: DataFormats/Scalers/src/Level1TriggerRates.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"

#include <iostream>
#include <cstdio>

Level1TriggerRates::Level1TriggerRates():
  version_(0),
  collectionTimeGeneral_(0,0),
  deltaT_(0),
  gtPartition0ResetsRate_(0.0),
  bunchCrossingErrorsRate_(0.0),
  gtPartition0TriggersRate_(0.0),
  gtPartition0EventsRate_(0.0),
  collectionTimeLumiSeg_(0,0),
  triggersPhysicsGeneratedFDLRate_(0.0),
  triggersPhysicsLostRate_(0.0),
  triggersPhysicsLostBeamActiveRate_(0.0),
  triggersPhysicsLostBeamInactiveRate_(0.0),
  l1AsPhysicsRate_(0.0),
  l1AsRandomRate_(0.0),
  l1AsTestRate_(0.0),
  l1AsCalibrationRate_(0.0),
  deadtimePercent_(0.0),
  deadtimeBeamActivePercent_(0.0),
  deadtimeBeamActiveTriggerRulesPercent_(0.0),
  deadtimeBeamActiveCalibrationPercent_(0.0),
  deadtimeBeamActivePrivateOrbitPercent_(0.0),
  deadtimeBeamActivePartitionControllerPercent_(0.0),
  deadtimeBeamActiveTimeSlotPercent_(0.0),
  gtAlgoCountsRate_(Level1TriggerScalers::nLevel1Triggers),
  gtTechCountsRate_(Level1TriggerScalers::nLevel1TestTriggers)
{ 
}

Level1TriggerRates::Level1TriggerRates(Level1TriggerScalers const& s)
{ 
  Level1TriggerRates();
}

Level1TriggerRates::Level1TriggerRates(Level1TriggerScalers const& s1,
			       Level1TriggerScalers const& s2)
{  
  Level1TriggerRates();

  const Level1TriggerScalers *t1 = &s1;
  const Level1TriggerScalers *t2 = &s2;

  // Choose the later sample to be t2

  computeRates(*t1,*t2);
}

Level1TriggerRates::~Level1TriggerRates() { } 


void Level1TriggerRates::computeRates(Level1TriggerScalers const& t1,
				      Level1TriggerScalers const& t2)
{
  version_ = t1.version();

  collectionTimeGeneral_.set_tv_sec(static_cast<long>(t1.collectionTimeGeneral().tv_sec));
  collectionTimeGeneral_.set_tv_nsec(t1.collectionTimeGeneral().tv_nsec);

  collectionTimeLumiSeg_.set_tv_sec(static_cast<long>(t1.collectionTimeLumiSeg().tv_sec));
  collectionTimeLumiSeg_.set_tv_nsec(t1.collectionTimeLumiSeg().tv_nsec);
}


/// Pretty-print operator for Level1TriggerRates
std::ostream& operator<<(std::ostream& s, const Level1TriggerRates& c) 
{
  s << "Level1TriggerRates Version: " << c.version() 
    << " Differential Rates in Hz, DeltaT: " <<
    c.deltaT() << " sec" << std::endl;
  //  char line[128];

  return s;
}
