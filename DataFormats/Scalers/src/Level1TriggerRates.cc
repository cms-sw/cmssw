/*
 *   File: DataFormats/Scalers/src/Level1TriggerRates.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"

#include <iostream>
#include <cstdio>

Level1TriggerRates::Level1TriggerRates():
  version_(0),
  collectionTime_(0,0),
  deltaNS_(0),
  deltaT_(0.0),
  gtTriggersRate_(0.0),
  gtEventsRate_(0.0),
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
  Level1TriggerRates(s,Level1TriggerScalers::firstShortLSRun);
}

Level1TriggerRates::Level1TriggerRates(Level1TriggerScalers const& s,
				       int runNumber)
{ 
  Level1TriggerRates();
  computeRates(s,runNumber);
}

Level1TriggerRates::Level1TriggerRates(Level1TriggerScalers const& s1,
				       Level1TriggerScalers const& s2)
{
  Level1TriggerRates(s1,s2,Level1TriggerScalers::firstShortLSRun);
}

Level1TriggerRates::Level1TriggerRates(Level1TriggerScalers const& s1,
				       Level1TriggerScalers const& s2,
				       int runNumber)
{  
  Level1TriggerRates();
  computeRates(s1,s2,runNumber);
}

Level1TriggerRates::~Level1TriggerRates() { } 


void Level1TriggerRates::computeRates(Level1TriggerScalers const& t)
{ computeRates(t,Level1TriggerScalers::firstShortLSRun);}

void Level1TriggerRates::computeRates(Level1TriggerScalers const& t,
				      int run)
{
  version_ = t.version();

  collectionTime_.set_tv_sec(static_cast<long>(t.collectionTime().tv_sec));
  collectionTime_.set_tv_nsec(t.collectionTime().tv_nsec);

  gtTriggersRate_ = t.gtTriggersRate();
  gtEventsRate_   = t.gtEventsRate();

  collectionTimeLumiSeg_.set_tv_sec(static_cast<long>(t.collectionTimeLumiSeg().tv_sec));
  collectionTimeLumiSeg_.set_tv_nsec(t.collectionTimeLumiSeg().tv_nsec);

  triggersPhysicsGeneratedFDLRate_ 
    = Level1TriggerScalers::rateLS(t.triggersPhysicsGeneratedFDL(),run);
  triggersPhysicsLostRate_ 
    = Level1TriggerScalers::rateLS(t.triggersPhysicsLost(),run);
  triggersPhysicsLostBeamActiveRate_ 
    = Level1TriggerScalers::rateLS(t.triggersPhysicsLostBeamActive(),run);
  triggersPhysicsLostBeamInactiveRate_ 
    = Level1TriggerScalers::rateLS(t.triggersPhysicsLostBeamInactive(),run);

  l1AsPhysicsRate_ = Level1TriggerScalers::rateLS(t.l1AsPhysics(),run);
  l1AsRandomRate_ = Level1TriggerScalers::rateLS(t.l1AsRandom(),run);
  l1AsTestRate_ = Level1TriggerScalers::rateLS(t.l1AsTest(),run);
  l1AsCalibrationRate_ = Level1TriggerScalers::rateLS(t.l1AsCalibration(),run);

  deadtimePercent_ = Level1TriggerScalers::percentLS(t.deadtime(),run);
  deadtimeBeamActivePercent_ =
    Level1TriggerScalers::percentLSActive(t.deadtimeBeamActive(),run);
  deadtimeBeamActiveTriggerRulesPercent_ =
    Level1TriggerScalers::percentLSActive(t.deadtimeBeamActiveTriggerRules(),run);
  deadtimeBeamActiveCalibrationPercent_ =
    Level1TriggerScalers::percentLSActive(t.deadtimeBeamActiveCalibration(),run);
  deadtimeBeamActivePrivateOrbitPercent_ =
    Level1TriggerScalers::percentLSActive(t.deadtimeBeamActivePrivateOrbit(),run);
  deadtimeBeamActivePartitionControllerPercent_ =
    Level1TriggerScalers::percentLSActive(t.deadtimeBeamActivePartitionController(),run);
  deadtimeBeamActiveTimeSlotPercent_ =
    Level1TriggerScalers::percentLSActive(t.deadtimeBeamActiveTimeSlot(),run);

  const std::vector<unsigned int> gtAlgoCounts = t.gtAlgoCounts();
  for ( std::vector<unsigned int>::const_iterator counts = gtAlgoCounts.begin();
	counts != gtAlgoCounts.end(); ++counts)
  {
    gtAlgoCountsRate_.push_back(Level1TriggerScalers::rateLS(*counts,run));
  }

  const std::vector<unsigned int> gtTechCounts = t.gtTechCounts();
  for ( std::vector<unsigned int>::const_iterator counts = gtTechCounts.begin();
	counts != gtTechCounts.end(); ++counts)
  {
    gtTechCountsRate_.push_back(Level1TriggerScalers::rateLS(*counts,run));
  }

  deltaNS_ = 0ULL;
  deltaT_ = 0.0;
}

void Level1TriggerRates::computeRates(Level1TriggerScalers const& t1,
				      Level1TriggerScalers const& t2)
{
  computeRates(t1,t2,Level1TriggerScalers::firstShortLSRun);
}

void Level1TriggerRates::computeRates(Level1TriggerScalers const& t1,
				      Level1TriggerScalers const& t2,
				      int run)
{
  computeRates(t1,run);

  unsigned long long zeit1 = 
    ( (unsigned long long)t1.collectionTime().tv_sec * 1000000000ULL)| 
    ( (unsigned long long)t1.collectionTime().tv_nsec );
  unsigned long long zeit2 = 
    ( (unsigned long long)t2.collectionTime().tv_sec * 1000000000ULL)| 
    ( (unsigned long long)t2.collectionTime().tv_nsec );

  deltaT_  = 0.0;
  deltaNS_ = 0ULL;
  if ( zeit2 > zeit1 ) 
  {
    deltaNS_ = zeit2 - zeit1;
    deltaT_  = ((double)deltaNS_) / 1.0E9;
    gtTriggersRate_ = 
      ((double)(t2.gtTriggers()-t1.gtTriggers()))/deltaT_;
    gtEventsRate_   = 
      ((double)(t2.gtEvents()-t1.gtEvents()))/deltaT_;
  }
}


/// Pretty-print operator for Level1TriggerRates
std::ostream& operator<<(std::ostream& s, const Level1TriggerRates& c) 
{
  constexpr size_t kLineBufferSize = 164;
  char line[kLineBufferSize];
  char zeitHeaven[128];
  struct tm * horaHeaven;

  s << "Level1TriggerRates Version: " << c.version() 
    << " Rates in Hz, DeltaT: ";

  if ( c.deltaNS() > 0 )
  {
    s << c.deltaT() << " sec" << std::endl;
  }
  else
  {
    s << "n/a" << std::endl;
  }

  struct timespec secondsToHeaven = c.collectionTime();
  horaHeaven = gmtime(&secondsToHeaven.tv_sec);
  strftime(zeitHeaven, sizeof(zeitHeaven), "%Y.%m.%d %H:%M:%S", horaHeaven);
  snprintf(line, kLineBufferSize, " CollectionTime:        %s.%9.9d" , 
           zeitHeaven, (int)secondsToHeaven.tv_nsec);
  s << line << std::endl;
  
  snprintf(line, kLineBufferSize, 
           " GtTriggersRate:                              %22.3f Hz",
           c.gtTriggersRate());
  s << line << std::endl;
  
  snprintf(line, kLineBufferSize,
           " GtEventsRate:                                %22.3f Hz",
           c.gtEventsRate());
  s << line << std::endl;

  secondsToHeaven = c.collectionTimeLumiSeg();
  horaHeaven = gmtime(&secondsToHeaven.tv_sec);
  strftime(zeitHeaven, sizeof(zeitHeaven), "%Y.%m.%d %H:%M:%S", horaHeaven);
  snprintf(line, kLineBufferSize, " CollectionTimeLumiSeg: %s.%9.9d" , 
	  zeitHeaven, (int)secondsToHeaven.tv_nsec);
  s << line << std::endl;
  
  snprintf(line, kLineBufferSize,
           " TriggersPhysicsGeneratedFDLRate:             %22.3f Hz",
           c.triggersPhysicsGeneratedFDLRate());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
           " TriggersPhysicsLostRate:                     %22.3f Hz",
           c.triggersPhysicsLostRate());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
           " TriggersPhysicsLostBeamActiveRate:           %22.3f Hz",
           c.triggersPhysicsLostBeamActiveRate());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
          " TriggersPhysicsLostBeamInactiveRate:         %22.3f Hz",
	  c.triggersPhysicsLostBeamInactiveRate());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
          " L1AsPhysicsRate:                             %22.3f Hz",
	  c.l1AsPhysicsRate());
  s << line << std::endl;

  snprintf(line, kLineBufferSize, 
           " L1AsRandomRate:                              %22.3f Hz",
           c.l1AsRandomRate());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
           " L1AsTestRate:                                %22.3f Hz",
           c.l1AsTestRate());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
           " L1AsCalibrationRate:                         %22.3f Hz",
           c.l1AsCalibrationRate());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
           " DeadtimePercent:                             %22.3f %%",
           c.deadtimePercent());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
           " DeadtimeBeamActivePercent:                   %22.3f %%",
           c.deadtimeBeamActivePercent());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
           " DeadtimeBeamActiveTriggerRulesPercent:       %22.3f %%",
           c.deadtimeBeamActiveTriggerRulesPercent());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
           " DeadtimeBeamActiveCalibrationPercent:        %22.3f %%",
           c.deadtimeBeamActiveCalibrationPercent());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
           " DeadtimeBeamActivePrivateOrbitPercent:       %22.3f %%",
           c.deadtimeBeamActivePrivateOrbitPercent());
  s << line << std::endl;

  snprintf(line, kLineBufferSize,
           " DeadtimeBeamActivePartitionControllerPercent:%22.3f %%",
           c.deadtimeBeamActivePartitionControllerPercent());
  s << line << std::endl;

  snprintf(line, kLineBufferSize, 
          " DeadtimeBeamActiveTimeSlotPercent:           %22.3f %%",
	  c.deadtimeBeamActiveTimeSlotPercent());
  s << line << std::endl;

  s << "Physics GtAlgoCountsRate, Hz" << std::endl;
  const std::vector<double> gtAlgoCountsRate = c.gtAlgoCountsRate();
  int length = gtAlgoCountsRate.size() / 4;
  for ( int i=0; i<length; i++)
  {
    snprintf(line, kLineBufferSize,
             " %3.3d: %12.3f  %3.3d: %12.3f  %3.3d: %12.3f  %3.3d: %12.3f",
             i,              gtAlgoCountsRate[i], 
             (i+length),     gtAlgoCountsRate[i+length], 
             (i+(length*2)), gtAlgoCountsRate[i+(length*2)], 
             (i+(length*3)), gtAlgoCountsRate[i+(length*3)]);
    s << line << std::endl;
  }

  s << "Test GtTechCountsRate, Hz" << std::endl;
  const std::vector<double> gtTechCountsRate = c.gtTechCountsRate();
  length = gtTechCountsRate.size() / 4;
  for ( int i=0; i<length; i++)
  {
    snprintf(line, kLineBufferSize,
             " %3.3d: %12.3f  %3.3d: %12.3f  %3.3d: %12.3f  %3.3d: %12.3f",
             i,              gtTechCountsRate[i], 
             (i+length),     gtTechCountsRate[i+length], 
             (i+(length*2)), gtTechCountsRate[i+(length*2)], 
             (i+(length*3)), gtTechCountsRate[i+(length*3)]);
    s << line << std::endl;
  }
  return s;
}
