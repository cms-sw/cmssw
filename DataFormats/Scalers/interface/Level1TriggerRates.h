/*
 *  File: DataFormats/Scalers/interface/Level1TriggerRates.h   (W.Badgett)
 *
 *  Various Level 1 Trigger Rates from the GT/TS
 *
 */

#ifndef DATAFORMATS_SCALERS_LEVEL1TRIGGERRATES_H
#define DATAFORMATS_SCALERS_LEVEL1TRIGGERRATES_H

#include "DataFormats/Scalers/interface/TimeSpec.h"

#include <ctime>
#include <iosfwd>
#include <string>
#include <vector>

/*! \file Level1TriggerRates.h
 * \Header file for Level 1 Global Trigger Rates
 * 
 * \author: William Badgett
 *
 */


/// \class Level1TriggerRates.h
/// \brief Persistable copy of Level1 Trigger Rates

class Level1TriggerScalers;

class Level1TriggerRates
{
 public:

#define BX_SPACING (double)25E-9

  Level1TriggerRates();
  Level1TriggerRates(Level1TriggerScalers const& s);
  Level1TriggerRates(Level1TriggerScalers const& s,
		     int runNumber);
  Level1TriggerRates(Level1TriggerScalers const& s1, 
		     Level1TriggerScalers const& s2);
  Level1TriggerRates(Level1TriggerScalers const& s1, 
		     Level1TriggerScalers const& s2,
		     int runNumber);
  virtual ~Level1TriggerRates();

  void computeRates(Level1TriggerScalers const& t1);
  void computeRates(Level1TriggerScalers const& t1, 
		    int runNumber);

  void computeRates(Level1TriggerScalers const& t1,
		    Level1TriggerScalers const& t2);
  void computeRates(Level1TriggerScalers const& t1,
		    Level1TriggerScalers const& t2,
		    int runNumber);

  /// name method
  std::string name() const { return "Level1TriggerRates"; }

  /// empty method (= false)
  bool empty() const { return false; }

  /// get the data

  int version() const { return(version_);}
  timespec collectionTime() { return(collectionTime_.get_timespec());}

  unsigned long long deltaNS()  const { return(deltaNS_);}
  double deltaT()               const { return(deltaT_);}

  double gtTriggersRate() const 
  { return(gtTriggersRate_);}

  double gtEventsRate() const 
  { return(gtEventsRate_);}

  timespec collectionTimeLumiSeg() 
  { return(collectionTimeLumiSeg_.get_timespec());}

  double triggersPhysicsGeneratedFDLRate() const 
  { return(triggersPhysicsGeneratedFDLRate_);}

  double triggersPhysicsLostRate() const 
  { return(triggersPhysicsLostRate_);}

  double triggersPhysicsLostBeamActiveRate() const 
  { return(triggersPhysicsLostBeamActiveRate_);}

  double triggersPhysicsLostBeamInactiveRate() const 
  { return(triggersPhysicsLostBeamInactiveRate_);}

  double l1AsPhysicsRate() const     { return(l1AsPhysicsRate_);}

  double l1AsRandomRate() const      { return(l1AsRandomRate_);}

  double l1AsTestRate() const        { return(l1AsTestRate_);}

  double l1AsCalibrationRate() const { return(l1AsCalibrationRate_);}

  double deadtimePercent() const     { return(deadtimePercent_);}

  double deadtimeBeamActivePercent() const 
  { return(deadtimeBeamActivePercent_);}

  double deadtimeBeamActiveTriggerRulesPercent() const 
  { return(deadtimeBeamActiveTriggerRulesPercent_);}

  double deadtimeBeamActiveCalibrationPercent() const 
  { return(deadtimeBeamActiveCalibrationPercent_);}

  double deadtimeBeamActivePrivateOrbitPercent() const 
  { return(deadtimeBeamActivePrivateOrbitPercent_);}

  double deadtimeBeamActivePartitionControllerPercent() const 
  { return(deadtimeBeamActivePartitionControllerPercent_);}

  double deadtimeBeamActiveTimeSlotPercent() const 
  { return(deadtimeBeamActiveTimeSlotPercent_);}

  timespec collectionTime() const 
  { return(collectionTime_.get_timespec());}

  timespec collectionTimeLumiSeg() const 
  { return(collectionTimeLumiSeg_.get_timespec());}

  std::vector<double> gtAlgoCountsRate() const { return(gtAlgoCountsRate_);}
  std::vector<double> gtTechCountsRate() const { return(gtTechCountsRate_);}

  /// equality operator
  int operator==(const Level1TriggerRates& e) const { return false; }

  /// inequality operator
  int operator!=(const Level1TriggerRates& e) const { return false; }

protected:

  int version_;

  TimeSpec collectionTime_;
  unsigned long long deltaNS_;
  double deltaT_;
  double gtTriggersRate_;
  double gtEventsRate_;

  TimeSpec collectionTimeLumiSeg_;
  double triggersPhysicsGeneratedFDLRate_;
  double triggersPhysicsLostRate_;
  double triggersPhysicsLostBeamActiveRate_;
  double triggersPhysicsLostBeamInactiveRate_;
  double l1AsPhysicsRate_;
  double l1AsRandomRate_;
  double l1AsTestRate_;
  double l1AsCalibrationRate_;
  double deadtimePercent_;
  double deadtimeBeamActivePercent_;
  double deadtimeBeamActiveTriggerRulesPercent_;
  double deadtimeBeamActiveCalibrationPercent_;
  double deadtimeBeamActivePrivateOrbitPercent_;
  double deadtimeBeamActivePartitionControllerPercent_;
  double deadtimeBeamActiveTimeSlotPercent_;

  std::vector<double> gtAlgoCountsRate_;
  std::vector<double> gtTechCountsRate_;
};


/// Pretty-print operator for Level1TriggerRates
std::ostream& operator<<(std::ostream& s, const Level1TriggerRates& c);

typedef std::vector<Level1TriggerRates> Level1TriggerRatesCollection;

#endif
