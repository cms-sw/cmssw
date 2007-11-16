
/*
 *   File: DataFormats/Scalers/src/L1TriggerScalers.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

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
  lostBunchCrossings_(0),
  lostTriggers_(0),
  lostTriggersActive_(0),
  triggers_(nL1Triggers)
{ 
  collectionTimeSummary_.tv_sec = 0;
  collectionTimeSummary_.tv_nsec = 0;
  collectionTimeDetails_.tv_sec = 0;
  collectionTimeDetails_.tv_nsec = 0;
}

L1TriggerScalers::L1TriggerScalers(const unsigned char * rawData)
{ 
  L1TriggerScalers();
  int version = ((int *)rawData)[0];
  if ( version == 1 )
  {
    version_ = version;
    ScalersEventRecordRaw_v1 * raw 
      = (struct ScalersEventRecordRaw_v1 *)rawData;

    collectionTimeSummary_.tv_sec 
      = raw->trig.collectionTimeSummary.tv_sec;
    collectionTimeSummary_.tv_nsec 
      = raw->trig.collectionTimeSummary.tv_nsec;
    triggerNumber_             = raw->trig.TRIGNR_;
    eventNumber_               = raw->trig.EVNR;
    physicsL1Accepts_          = raw->trig.PHYS_L1A;
    physicsL1AcceptsRaw_       = raw->trig.FINOR_;
    randomL1Accepts_           = raw->trig.RNDM_L1A_;
    calibrationL1Accepts_      = raw->trig.CAL_L1A_;
    techTrig_                  = raw->trig.TECHTRIG_;
    orbitNumber_               = raw->trig.ORBITNR;
    numberResets_              = raw->trig.NR_RESETS_;
    deadTime_                  = raw->trig.DEADT_;
    deadTimeActive_            = raw->trig.DEADT_A;
    deadTimeActiveCalibration_ = raw->trig.DEADT_CALIBR_A;
    deadTimeActivePrivate_     = raw->trig.DEADT_PRIV_A;
    deadTimeActivePartition_   = raw->trig.DEADT_PSTATUS_A;
    deadTimeActiveThrottle_    = raw->trig.DEADT_THROTTLE_A;
    lostBunchCrossings_        = raw->trig.LOST_BC_;
    lostTriggers_              = raw->trig.LOST_TRIG_;
    lostTriggersActive_        = raw->trig.LOST_TRIG_A;

    collectionTimeDetails_.tv_sec 
      = raw->trig.collectionTimeDetails.tv_sec;
    collectionTimeDetails_.tv_nsec 
      = raw->trig.collectionTimeDetails.tv_nsec;

    for ( int i=0; i<ScalersRaw::N_L1_TRIGGERS_v1; i++)
    { triggers_[i] = raw->trig.RATE_ALGO[i];  }
  }
}

L1TriggerScalers::~L1TriggerScalers() { } 


/// Pretty-print operator for L1TriggerScalers
std::ostream& operator<<(std::ostream& s, const L1TriggerScalers& c) 
{
  s << " L1TriggerScalers: ";
  return s;
}
