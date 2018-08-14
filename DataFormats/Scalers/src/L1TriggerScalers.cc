
/*
 *   File: DataFormats/Scalers/src/L1TriggerScalers.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

#include <iostream>
#include <cstdio>

L1TriggerScalers::L1TriggerScalers():
  version_(0),
  collectionTimeSpecial_(0,0),
  orbitNumber_(0),
  luminositySection_(0),
  bunchCrossingErrors_(0),
  collectionTimeSummary_(0,0),
  triggerNumber_(0),
  eventNumber_(0),
  finalTriggersDistributed_(0),
  calibrationTriggers_(0),
  randomTriggers_(0),
  totalTestTriggers_(0),
  finalTriggersGenerated_(0),
  finalTriggersInvalidBC_(0),
  deadTime_(0),
  lostFinalTriggers_(0),
  deadTimeActive_(0),
  lostFinalTriggersActive_(0),
  deadTimeActivePrivate_(0),
  deadTimeActivePartition_(0),
  deadTimeActiveThrottle_(0),
  deadTimeActiveCalibration_(0),
  deadTimeActiveTimeSlot_(0),
  numberResets_(0),
  collectionTimeDetails_(0,0),
  triggers_(nL1Triggers),
  testTriggers_(nL1TestTriggers)
{ 
}

L1TriggerScalers::L1TriggerScalers(const unsigned char * rawData)
{ 
  L1TriggerScalers();

  struct ScalersEventRecordRaw_v1 const * raw 
    = reinterpret_cast<struct ScalersEventRecordRaw_v1 const *>(rawData);

  trigType_     = ( raw->header >> 56 ) &        0xFULL;
  eventID_      = ( raw->header >> 32 ) & 0x00FFFFFFULL;
  sourceID_     = ( raw->header >>  8 ) & 0x00000FFFULL;
  bunchNumber_  = ( raw->header >> 20 ) &      0xFFFULL;

  version_ = raw->version;
  if ( ( version_ == 1 ) || ( version_ == 2 ) )
  {
    collectionTimeSpecial_.set_tv_sec( static_cast<long>(
      raw->trig.collectionTimeSpecial_sec));
    collectionTimeSpecial_.set_tv_nsec( 
      raw->trig.collectionTimeSpecial_nsec);
    orbitNumber_               = raw->trig.ORBIT_NUMBER;
    luminositySection_         = raw->trig.LUMINOSITY_SEGMENT;
    bunchCrossingErrors_       = raw->trig.BC_ERRORS;

    collectionTimeSummary_.set_tv_sec( static_cast<long>(
      raw->trig.collectionTimeSummary_sec));
    collectionTimeSummary_.set_tv_nsec( 
      raw->trig.collectionTimeSummary_nsec);

    triggerNumber_             = raw->trig.TRIGGER_NR;
    eventNumber_               = raw->trig.EVENT_NR;
    finalTriggersDistributed_  = raw->trig.FINOR_DISTRIBUTED;
    calibrationTriggers_       = raw->trig.CAL_TRIGGER;
    randomTriggers_            = raw->trig.RANDOM_TRIGGER;
    totalTestTriggers_         = raw->trig.TEST_TRIGGER;

    finalTriggersGenerated_    = raw->trig.FINOR_GENERATED;
    finalTriggersInvalidBC_    = raw->trig.FINOR_IN_INVALID_BC;

    deadTime_                  = raw->trig.DEADTIME;
    lostFinalTriggers_         = raw->trig.LOST_FINOR;
    deadTimeActive_            = raw->trig.DEADTIMEA;
    lostFinalTriggersActive_   = raw->trig.LOST_FINORA;
    deadTimeActivePrivate_     = raw->trig.PRIV_DEADTIMEA;
    deadTimeActivePartition_   = raw->trig.PTCSTATUS_DEADTIMEA;
    deadTimeActiveThrottle_    = raw->trig.THROTTLE_DEADTIMEA;
    deadTimeActiveCalibration_ = raw->trig.CALIBRATION_DEADTIMEA;
    deadTimeActiveTimeSlot_    = raw->trig.TIMESLOT_DEADTIMEA;
    numberResets_              = raw->trig.NR_OF_RESETS;

    collectionTimeDetails_.set_tv_sec( static_cast<long>(
      raw->trig.collectionTimeDetails_sec));
    collectionTimeDetails_.set_tv_nsec(
      raw->trig.collectionTimeDetails_nsec);

    for ( int i=0; i<ScalersRaw::N_L1_TRIGGERS_v1; i++)
    { triggers_.push_back( raw->trig.ALGO_RATE[i]);}

    for ( int i=0; i<ScalersRaw::N_L1_TEST_TRIGGERS_v1; i++)
    { testTriggers_.push_back( raw->trig.TEST_RATE[i]);}
  }
}

L1TriggerScalers::~L1TriggerScalers() { } 


/// Pretty-print operator for L1TriggerScalers
std::ostream& operator<<(std::ostream& s,L1TriggerScalers const &c) 
{
  s << "L1TriggerScalers    Version:" << c.version() <<
    "   SourceID: " << c.sourceID() << std::endl;
  constexpr size_t kLineBufferSize = 164;
  char line[kLineBufferSize];
  char zeitHeaven[128];
  char zeitHell[128];
  char zeitLimbo[128];
  struct tm * horaHeaven;
  struct tm * horaHell;
  struct tm * horaLimbo;

  sprintf(line, " TrigType: %d   EventID: %d    BunchNumber: %d", 
	  c.trigType(), c.eventID(), c.bunchNumber());
  s << line << std::endl;

  timespec secondsToHeaven = c.collectionTimeSummary();
  horaHeaven = gmtime(&secondsToHeaven.tv_sec);
  strftime(zeitHeaven, sizeof(zeitHeaven), "%Y.%m.%d %H:%M:%S", horaHeaven);
  snprintf(line, kLineBufferSize, " CollectionTimeSummary: %s.%9.9d" , 
	  zeitHeaven, (int)secondsToHeaven.tv_nsec);
  s << line << std::endl;

  timespec secondsToHell= c.collectionTimeSpecial();
  horaHell = gmtime(&secondsToHell.tv_sec);
  strftime(zeitHell, sizeof(zeitHell), "%Y.%m.%d %H:%M:%S", horaHell);
  snprintf(line, kLineBufferSize, " CollectionTimeSpecial: %s.%9.9d" , 
	  zeitHell, (int)secondsToHell.tv_nsec);
  s << line << std::endl;

  timespec secondsToLimbo= c.collectionTimeDetails();
  horaLimbo = gmtime(&secondsToLimbo.tv_sec);
  strftime(zeitLimbo, sizeof(zeitLimbo), "%Y.%m.%d %H:%M:%S", horaLimbo);
  snprintf(line, kLineBufferSize, " CollectionTimeDetails: %s.%9.9d" , 
	  zeitLimbo, (int)secondsToLimbo.tv_nsec);
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
	  " LuminositySection: %15d  BunchCrossingErrors:      %15d",
	  c.luminositySection(), c.bunchCrossingErrors());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
	  " TriggerNumber:     %15d  EventNumber:              %15d",
	  c.triggerNumber(), c.eventNumber());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
	  " TriggersDistributed:    %10d  TriggersGenerated:        %15d",
	  c.finalTriggersDistributed(), 
	  c.finalTriggersGenerated());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
	  " TriggersInvalidBC: %15d  CalibrationTriggers:      %15d",
	  c.finalTriggersInvalidBC(), c.calibrationTriggers());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
	  " TestTriggers:      %15d  RandomTriggers:           %15d",
	  c.totalTestTriggers(), c.randomTriggers());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
	  " DeadTime:          %15d  DeadTimeActiveTimeSlot:   %15ld",
	  c.numberResets(), (long int)c.deadTime());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
	  " DeadTimeActive:    %15ld  DeadTimeActiveCalibration:%15ld",
	  (long int)c.deadTimeActive(), 
	  (long int)c.deadTimeActiveCalibration());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
	  " LostTriggers:      %15ld  DeadTimeActivePartition:  %15ld",
	  (long int)c.lostFinalTriggers(), 
	  (long int)c.deadTimeActivePartition());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
	  " LostTriggersActive:%15ld  DeadTimeActiveThrottle:   %15ld",
	  (long int)c.lostFinalTriggersActive(),
	  (long int)c.deadTimeActiveThrottle());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
	  " NumberResets:      %15d  DeadTimeActivePrivate:    %15ld",
	  c.numberResets(),
	  (long int)c.deadTimeActivePrivate());
  s << line << std::endl;

  s << "Physics Triggers" << std::endl;
  std::vector<unsigned int> triggers = c.triggers();
  int length = triggers.size() / 4;
  for ( int i=0; i<length; i++)
  {
    snprintf(line, kLineBufferSize,
             " %3.3d: %10d    %3.3d: %10d    %3.3d: %10d    %3.3d: %10d",
	    i,              triggers[i], 
	    (i+length),     triggers[i+length], 
	    (i+(length*2)), triggers[i+(length*2)], 
	    (i+(length*3)), triggers[i+(length*3)]);
    s << line << std::endl;
  }

  s << "Test Triggers" << std::endl;
  std::vector<unsigned int> testTriggers = c.testTriggers();
  length = testTriggers.size() / 4;
  for ( int i=0; i<length; i++)
  {
    snprintf(line, kLineBufferSize,
             " %3.3d: %10d    %3.3d: %10d    %3.3d: %10d    %3.3d: %10d",
	    i,              testTriggers[i], 
	    (i+length),     testTriggers[i+length], 
	    (i+(length*2)), testTriggers[i+(length*2)], 
	    (i+(length*3)), testTriggers[i+(length*3)]);
    s << line << std::endl;
  }
  return s;
}
