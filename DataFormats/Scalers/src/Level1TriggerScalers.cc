
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
  collectionTime_(0,0),
  lumiSegmentNr_(0),
  lumiSegmentOrbits_(0),
  orbitNr_(0),
  gtResets_(0),
  bunchCrossingErrors_(0),
  gtTriggers_(0),
  gtEvents_(0),
  gtTriggersRate_((float)0.0),
  gtEventsRate_((float)0.0),
  prescaleIndexAlgo_(0),
  prescaleIndexTech_(0),
  collectionTimeLumiSeg_(0,0),
  lumiSegmentNrLumiSeg_(0),
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
  gtAlgoCounts_(nLevel1Triggers),
  gtTechCounts_(nLevel1TestTriggers)
{ 
}

Level1TriggerScalers::Level1TriggerScalers(const unsigned char * rawData)
{ 
  Level1TriggerScalers();

  struct ScalersEventRecordRaw_v3 * raw 
    = (struct ScalersEventRecordRaw_v3 *)rawData;

  trigType_     = ( raw->header >> 56 ) &        0xFULL;
  eventID_      = ( raw->header >> 32 ) & 0x00FFFFFFULL;
  sourceID_     = ( raw->header >>  8 ) & 0x00000FFFULL;
  bunchNumber_  = ( raw->header >> 20 ) &      0xFFFULL;

  version_      = raw->version;
  if ( version_ >= 3 )
  {
    collectionTime_.set_tv_sec( static_cast<long>(
      raw->trig.collectionTime_sec));
    collectionTime_.set_tv_nsec( 
      raw->trig.collectionTime_nsec);

    lumiSegmentNr_        = raw->trig.lumiSegmentNr;
    lumiSegmentOrbits_    = raw->trig.lumiSegmentOrbits;
    orbitNr_              = raw->trig.orbitNr;
    gtResets_             = raw->trig.gtResets;
    bunchCrossingErrors_  = raw->trig.bunchCrossingErrors;
    gtTriggers_           = raw->trig.gtTriggers;
    gtEvents_             = raw->trig.gtEvents;
    gtTriggersRate_       = raw->trig.gtTriggersRate;
    gtEventsRate_         = raw->trig.gtEventsRate;
    prescaleIndexAlgo_    = raw->trig.prescaleIndexAlgo;
    prescaleIndexTech_    = raw->trig.prescaleIndexTech;

    collectionTimeLumiSeg_.set_tv_sec( static_cast<long>(
      raw->trig.collectionTimeLumiSeg_sec));
    collectionTimeLumiSeg_.set_tv_nsec( 
      raw->trig.collectionTimeLumiSeg_nsec);

    lumiSegmentNrLumiSeg_           = raw->trig.lumiSegmentNrLumiSeg;
    triggersPhysicsGeneratedFDL_    = raw->trig.triggersPhysicsGeneratedFDL;
    triggersPhysicsLost_            = raw->trig.triggersPhysicsLost;
    triggersPhysicsLostBeamActive_  = raw->trig.triggersPhysicsLostBeamActive;
    triggersPhysicsLostBeamInactive_ = 
      raw->trig.triggersPhysicsLostBeamInactive;

    l1AsPhysics_                    = raw->trig.l1AsPhysics;
    l1AsRandom_                     = raw->trig.l1AsRandom;
    l1AsTest_                       = raw->trig.l1AsTest;
    l1AsCalibration_                = raw->trig.l1AsCalibration;
    deadtime_                       = raw->trig.deadtime;
    deadtimeBeamActive_             = raw->trig.deadtimeBeamActive;
    deadtimeBeamActiveTriggerRules_ = raw->trig.deadtimeBeamActiveTriggerRules;
    deadtimeBeamActiveCalibration_  = raw->trig.deadtimeBeamActiveCalibration;
    deadtimeBeamActivePrivateOrbit_ = raw->trig.deadtimeBeamActivePrivateOrbit;
    deadtimeBeamActivePartitionController_ = 
      raw->trig.deadtimeBeamActivePartitionController;
    deadtimeBeamActiveTimeSlot_ = raw->trig.deadtimeBeamActiveTimeSlot;

    for ( int i=0; i<ScalersRaw::N_L1_TRIGGERS_v1; i++)
    { gtAlgoCounts_.push_back( raw->trig.gtAlgoCounts[i]);}

    for ( int i=0; i<ScalersRaw::N_L1_TEST_TRIGGERS_v1; i++)
    { gtTechCounts_.push_back( raw->trig.gtTechCounts[i]);}
  }
}

Level1TriggerScalers::~Level1TriggerScalers() { } 

double Level1TriggerScalers::rateLS(unsigned int counts)
{ 
  unsigned long long counts64 = (unsigned long long)counts;
  return(rateLS(counts64));
}

double Level1TriggerScalers::rateLS(unsigned long long counts)
{ 
  double rate = ((double)counts) / 93.4281216;
  return(rate);
}

double Level1TriggerScalers::percentLS(unsigned long long counts)
{ 
  double percent = ((double)counts) / 37371248.64;
  if ( percent > 100.0000 ) { percent = 100.0;}
  return(percent);
}

double Level1TriggerScalers::percentLSActive(unsigned long long counts)
{ 
  double percent = ((double)counts) / 29444014.08;
  if ( percent > 100.0000 ) { percent = 100.0;}
  return(percent);
}

/// Pretty-print operator for Level1TriggerScalers
std::ostream& operator<<(std::ostream& s,Level1TriggerScalers const &c) 
{
  s << "Level1TriggerScalers    Version:" << c.version() <<
    "   SourceID: " << c.sourceID() << std::endl;
  char line[128];
  char zeitHeaven[128];
  struct tm * horaHeaven;

  sprintf(line, " TrigType: %d   EventID: %d    BunchNumber: %d", 
	  c.trigType(), c.eventID(), c.bunchNumber());
  s << line << std::endl;

  struct timespec secondsToHeaven = c.collectionTime();
  horaHeaven = gmtime(&secondsToHeaven.tv_sec);
  strftime(zeitHeaven, sizeof(zeitHeaven), "%Y.%m.%d %H:%M:%S", horaHeaven);
  sprintf(line, " CollectionTime:        %s.%9.9d" , 
	  zeitHeaven, (int)secondsToHeaven.tv_nsec);
  s << line << std::endl;

  sprintf(line,
	  " LumiSegmentNr:        %10u   LumiSegmentOrbits:     %10u",
	  c.lumiSegmentNr(), c.lumiSegmentOrbits());
  s << line << std::endl;

  sprintf(line,
	  " LumiSegmentNrLumiSeg: %10u   OrbitNr:               %10u ",
	  c.lumiSegmentNrLumiSeg(),  c.orbitNr());
  s << line << std::endl;

  sprintf(line,
	  " GtResets:             %10u   BunchCrossingErrors:   %10u",
	  c.gtResets(), c.bunchCrossingErrors());
  s << line << std::endl;

  sprintf(line,
	  " PrescaleIndexAlgo:    %10d   PrescaleIndexTech:     %10d",
	  c.prescaleIndexAlgo(), c.prescaleIndexTech());
  s << line << std::endl;

  sprintf(line, " GtTriggers:                      %20llu %22.3f Hz", 
	  c.gtTriggers(), c.gtTriggersRate());
  s << line << std::endl;

  sprintf(line, " GtEvents:                        %20llu %22.3f Hz", 
	  c.gtEvents(), c.gtEventsRate());
  s << line << std::endl;

  secondsToHeaven = c.collectionTimeLumiSeg();
  horaHeaven = gmtime(&secondsToHeaven.tv_sec);
  strftime(zeitHeaven, sizeof(zeitHeaven), "%Y.%m.%d %H:%M:%S", horaHeaven);
  sprintf(line, " CollectionTimeLumiSeg: %s.%9.9d" , 
	  zeitHeaven, (int)secondsToHeaven.tv_nsec);
  s << line << std::endl;


  sprintf(line, " TriggersPhysicsGeneratedFDL:     %20llu %22.3f Hz",
	  c.triggersPhysicsGeneratedFDL(),
	  Level1TriggerScalers::rateLS(c.triggersPhysicsGeneratedFDL()));
  s << line << std::endl;

  sprintf(line, " TriggersPhysicsLost:             %20llu %22.3f Hz",
	  c.triggersPhysicsLost(),
	  Level1TriggerScalers::rateLS(c.triggersPhysicsLost()));
  s << line << std::endl;

  sprintf(line, " TriggersPhysicsLostBeamActive:   %20llu %22.3f Hz",
	  c.triggersPhysicsLostBeamActive(),
	  Level1TriggerScalers::rateLS(c.triggersPhysicsLostBeamActive()));
  s << line << std::endl;

  sprintf(line, " TriggersPhysicsLostBeamInactive: %20llu %22.3f Hz",
	  c.triggersPhysicsLostBeamInactive(),
	  Level1TriggerScalers::rateLS(c.triggersPhysicsLostBeamInactive()));
  s << line << std::endl;

  sprintf(line, " L1AsPhysics:                     %20llu %22.3f Hz",
	  c.l1AsPhysics(),
	  Level1TriggerScalers::rateLS(c.l1AsPhysics()));
  s << line << std::endl;

  sprintf(line, " L1AsRandom:                      %20llu %22.3f Hz",
	  c.l1AsRandom(),
	  Level1TriggerScalers::rateLS(c.l1AsRandom()));
  s << line << std::endl;

  sprintf(line, " L1AsTest:                        %20llu %22.3f Hz",
	  c.l1AsTest(),
	  Level1TriggerScalers::rateLS(c.l1AsTest()));
  s << line << std::endl;

  sprintf(line, " L1AsCalibration:                 %20llu %22.3f Hz",
	  c.l1AsCalibration(),
	  Level1TriggerScalers::rateLS(c.l1AsCalibration()));
  s << line << std::endl;

  sprintf(line, " Deadtime:                             %20llu %17.3f%%",
	  c.deadtime(),
	  Level1TriggerScalers::percentLS(c.deadtime()));
  s << line << std::endl;

  sprintf(line, " DeadtimeBeamActive:                   %20llu %17.3f%%",
	  c.deadtimeBeamActive(),
	  Level1TriggerScalers::percentLSActive(c.deadtimeBeamActive()));
  s << line << std::endl;

  sprintf(line, " DeadtimeBeamActiveTriggerRules:       %20llu %17.3f%%",
	  c.deadtimeBeamActiveTriggerRules(),
	  Level1TriggerScalers::percentLSActive(c.deadtimeBeamActiveTriggerRules()));
  s << line << std::endl;

  sprintf(line, " DeadtimeBeamActiveCalibration:        %20llu %17.3f%%",
	  c.deadtimeBeamActiveCalibration(),
	  Level1TriggerScalers::percentLSActive(c.deadtimeBeamActiveCalibration()));
  s << line << std::endl;

  sprintf(line, " DeadtimeBeamActivePrivateOrbit:       %20llu %17.3f%%",
	  c.deadtimeBeamActivePrivateOrbit(),
	  Level1TriggerScalers::percentLSActive(c.deadtimeBeamActivePrivateOrbit()));
  s << line << std::endl;

  sprintf(line, " DeadtimeBeamActivePartitionController:%20llu %17.3f%%",
	  c.deadtimeBeamActivePartitionController(),
	  Level1TriggerScalers::percentLSActive(c.deadtimeBeamActivePartitionController()));
  s << line << std::endl;

  sprintf(line, " DeadtimeBeamActiveTimeSlot:           %20llu %17.3f%%",
	  c.deadtimeBeamActiveTimeSlot(),
	  Level1TriggerScalers::percentLSActive(c.deadtimeBeamActiveTimeSlot()));
  s << line << std::endl;

  s << "Physics GtAlgoCounts" << std::endl;
  const std::vector<unsigned int> gtAlgoCounts = c.gtAlgoCounts();
  int length = gtAlgoCounts.size() / 4;
  for ( int i=0; i<length; i++)
  {
    sprintf(line," %3.3d: %10u    %3.3d: %10u    %3.3d: %10u    %3.3d: %10u",
	    i,              gtAlgoCounts[i], 
	    (i+length),     gtAlgoCounts[i+length], 
	    (i+(length*2)), gtAlgoCounts[i+(length*2)], 
	    (i+(length*3)), gtAlgoCounts[i+(length*3)]);
    s << line << std::endl;
  }

  s << "Test GtTechCounts" << std::endl;
  const std::vector<unsigned int> gtTechCounts = c.gtTechCounts();
  length = gtTechCounts.size() / 4;
  for ( int i=0; i<length; i++)
  {
    sprintf(line," %3.3d: %10u    %3.3d: %10u    %3.3d: %10u    %3.3d: %10u",
	    i,              gtTechCounts[i], 
	    (i+length),     gtTechCounts[i+length], 
	    (i+(length*2)), gtTechCounts[i+(length*2)], 
	    (i+(length*3)), gtTechCounts[i+(length*3)]);
    s << line << std::endl;
  }
  return s;
}
