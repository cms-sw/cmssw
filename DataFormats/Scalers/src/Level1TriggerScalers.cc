
/*
 *   File: DataFormats/Scalers/src/Level1TriggerScalers.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

#include <iostream>
#include <time.h>

Level1TriggerScalers::Level1TriggerScalers():
  version_(0),
  trigType_(0),
  eventID_(0),
  sourceID_(0),
  bunchNumber_(0),
  collectionTimeGeneral_(0,0),
  lumiSegmentNr_(0),
  lumiSegmentOrbits_(0),
  orbitNr_(0),
  gtPartition0Resets_(0),
  bunchCrossingErrors_(0),
  gtPartition0Triggers_(0),
  gtPartition0Events_(0),
  prescaleIndexAlgo_(0),
  prescaleIndexTech_(0),
  collectionTimeLumiSeg_(0,0),
  triggersPhysicsGeneratedFDL_(0),
  triggersPhysicsLost_(0),
  triggersPhysicsLostBeamActive_(0),
  triggersPhysicsLostBeamInactive_(0),
  l1AsPhysics_(0),
  l1AsRandom_(0),
  l1AsTest_(0),
  l1AsCalibration_(0),
  deadtime_(0),
  deadtimeBeamActive_(0),
  deadtimeBeamActiveTriggerRules_(0),
  deadtimeBeamActiveCalibration_(0),
  deadtimeBeamActivePrivateOrbit_(0),
  deadtimeBeamActivePartitionController_(0),
  deadtimeBeamActiveTimeSlot_(0),
  gtAlgoCounts_(nL1Triggers),
  gtTechCounts_(nL1TestTriggers),
{ 
}

Level1TriggerScalers::Level1TriggerScalers(const unsigned char * rawData)
{ 
  Level1TriggerScalers();

  ScalersEventRecordRaw_v3 * raw 
    = (struct ScalersEventRecordRaw_v3 *)rawData;

  trigType_     = ( raw->header >> 56 ) &        0xFULL;
  eventID_      = ( raw->header >> 32 ) & 0x00FFFFFFULL;
  sourceID_     = ( raw->header >>  8 ) & 0x00000FFFULL;
  bunchNumber_  = ( raw->header >> 20 ) &      0xFFFULL;

  version_ = raw->version;
  if ( version_ >= 3 )
  {
    collectionTimeGeneral_.set_tv_sec( static_cast<long>(
      raw->trig.collectionTimeGeneral_sec));
    collectionTimeGeneral_.set_tv_nsec( 
      raw->trig.collectionTimeGeneral_nsec);

    collectionTimeLumiSeg_.set_tv_sec( static_cast<long>(
      raw->trig.collectionTimeLumiSeg_sec));
    collectionTimeLumiSeg_.set_tv_nsec( 
      raw->trig.collectionTimeLumiSeg_nsec);

    for ( int i=0; i<ScalersRaw::N_L1_TRIGGERS_v1; i++)
    { triggers_.push_back( raw->trig.ALGO_RATE[i]);}

    for ( int i=0; i<ScalersRaw::N_L1_TEST_TRIGGERS_v1; i++)
    { testTriggers_.push_back( raw->trig.TEST_RATE[i]);}
  }
}

Level1TriggerScalers::~Level1TriggerScalers() { } 


/// Pretty-print operator for Level1TriggerScalers
std::ostream& operator<<(std::ostream& s,Level1TriggerScalers const &c) 
{
  s << "Level1TriggerScalers    Version:" << c.version() <<
    "   SourceID: " << c.sourceID() << std::endl;
  char line[128];
  char zeitHeaven[128];
  char zeitHell[128];
  struct tm * horaHeaven;
  struct tm * horaHell;

  sprintf(line, " TrigType: %d   EventID: %d    BunchNumber: %d", 
	  c.trigType(), c.eventID(), c.bunchNumber());
  s << line << std::endl;

  struct timespec secondsToHeaven = c.collectionTimeSummary();
  horaHeaven = gmtime(&secondsToHeaven.tv_sec);
  strftime(zeitHeaven, sizeof(zeitHeaven), "%Y.%m.%d %H:%M:%S", horaHeaven);
  sprintf(line, " CollectionTimeSummary: %s.%9.9d" , 
	  zeitHeaven, (int)secondsToHeaven.tv_nsec);
  s << line << std::endl;

  struct timespec secondsToHell= c.collectionTimeDetails();
  horaHell = gmtime(&secondsToHell.tv_sec);
  strftime(zeitHell, sizeof(zeitHell), "%Y.%m.%d %H:%M:%S", horaHell);
  sprintf(line, " CollectionTimeDetails: %s.%9.9d" , 
	  zeitHell, (int)secondsToHell.tv_nsec);
  s << line << std::endl;

  sprintf(line,
	  " LuminositySection: %15d  BunchCrossingErrors:      %15d",
	  c.luminositySection(), c.bunchCrossingErrors());
  s << line << std::endl;

  sprintf(line,
	  " TriggerNumber:     %15d  EventNumber:              %15d",
	  c.triggerNumber(), c.eventNumber());
  s << line << std::endl;

  sprintf(line,
	  " TriggersDistributed:    %10d  TriggersGenerated:        %15d",
	  c.finalTriggersDistributed(), 
	  c.finalTriggersGenerated());
  s << line << std::endl;

  sprintf(line,
	  " TriggersInvalidBC: %15d  CalibrationTriggers:      %15d",
	  c.finalTriggersInvalidBC(), c.calibrationTriggers());
  s << line << std::endl;

  sprintf(line,
	  " TestTriggers:      %15d  RandomTriggers:           %15d",
	  c.totalTestTriggers(), c.randomTriggers());
  s << line << std::endl;

  sprintf(line,
	  " DeadTime:          %15d  DeadTimeActiveTimeSlot:   %15ld",
	  c.numberResets(), (long int)c.deadTime());
  s << line << std::endl;

  sprintf(line,
	  " DeadTimeActive:    %15ld  DeadTimeActiveCalibration:%15ld",
	  (long int)c.deadTimeActive(), 
	  (long int)c.deadTimeActiveCalibration());
  s << line << std::endl;

  sprintf(line,
	  " LostTriggers:      %15ld  DeadTimeActivePartition:  %15ld",
	  (long int)c.lostFinalTriggers(), 
	  (long int)c.deadTimeActivePartition());
  s << line << std::endl;

  sprintf(line,
	  " LostTriggersActive:%15ld  DeadTimeActiveThrottle:   %15ld",
	  (long int)c.lostFinalTriggersActive(),
	  (long int)c.deadTimeActiveThrottle());
  s << line << std::endl;

  sprintf(line,
	  " NumberResets:      %15d  DeadTimeActivePrivate:    %15ld",
	  c.numberResets(),
	  (long int)c.deadTimeActivePrivate());
  s << line << std::endl;

  s << "Physics Triggers" << std::endl;
  std::vector<unsigned int> triggers = c.triggers();
  int length = triggers.size() / 4;
  for ( int i=0; i<length; i++)
  {
    sprintf(line," %3.3d: %10d    %3.3d: %10d    %3.3d: %10d    %3.3d: %10d",
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
    sprintf(line," %3.3d: %10d    %3.3d: %10d    %3.3d: %10d    %3.3d: %10d",
	    i,              testTriggers[i], 
	    (i+length),     testTriggers[i+length], 
	    (i+(length*2)), testTriggers[i+(length*2)], 
	    (i+(length*3)), testTriggers[i+(length*3)]);
    s << line << std::endl;
  }
  return s;
}
