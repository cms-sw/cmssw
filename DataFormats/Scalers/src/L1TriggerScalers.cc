
/*
 *   File: DataFormats/Scalers/src/L1TriggerScalers.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/L1TriggerScalers.h"

L1TriggerScalers::L1TriggerScalers():
  version_(0),
  triggerNumber_(0),
  eventNumber_(0),
  physicsL1Accepts_(0),
  physicsL1AcceptsRaw_(0),
  randomL1Accepts_(0),
  calibrationL1Accepts_(0),
  techTrig_(0),
  orbitNumber_(0),
  numberResets_(0),
  deadTime_(0),
  deadTimeActive_(0),
  deadTimeActiveCalibration_(0),
  deadTimeActivePrivate_(0),
  deadTimeActivePartition_(0),
  deadTimeActiveThrottle_(0),
  triggers_(nL1Triggers)
{ 
   collectionTimeSummary_.tv_sec = 0;
   collectionTimeSummary_.tv_nsec = 0;
   collectionTimeDetails_.tv_sec = 0;
   collectionTimeDetails_.tv_nsec = 0;
}

L1TriggerScalers::L1TriggerScalers(const unsigned char * rawData)
{ }

L1TriggerScalers::~L1TriggerScalers() { } 


/// Pretty-print operator for L1TriggerScalers
std::ostream& operator<<(std::ostream& s, const L1TriggerScalers& c) 
{
  s << " L1TriggerScalers: ";
  return s;
}
