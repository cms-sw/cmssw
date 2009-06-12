/*
 *  File: DataFormats/Scalers/interface/L1TriggerRates.h   (W.Badgett)
 *
 *  Various Level 1 Trigger Rates from the GT/TS
 *
 */

#ifndef DATAFORMATS_SCALERS_L1TRIGGERRATES_H
#define DATAFORMATS_SCALERS_L1TRIGGERRATES_H

#include "DataFormats/Scalers/interface/TimeSpec.h"

#include <ctime>
#include <iosfwd>
#include <string>
#include <vector>

/*! \file L1TriggerRates.h
 * \Header file for Level 1 Global Trigger Rates
 * 
 * \author: William Badgett
 *
 */


/// \class L1TriggerRates.h
/// \brief Persistable copy of L1 Trigger Rates

class L1TriggerScalers;

class L1TriggerRates
{
 public:

  enum
  {
    N_BX = 3654,
    N_BX_ACTIVE = 2808
  };

#define BX_SPACING (double)25E-9

  L1TriggerRates();
  L1TriggerRates(L1TriggerScalers const& s);
  L1TriggerRates(L1TriggerScalers const& s1, L1TriggerScalers const& s2);
  virtual ~L1TriggerRates();

  void computeRunRates(L1TriggerScalers const& t);
  void computeRates(L1TriggerScalers const& t1,
		    L1TriggerScalers const& t2);

  /// name method
  std::string name() const { return "L1TriggerRates"; }

  /// empty method (= false)
  bool empty() const { return false; }

  /// get the data

  int version() const { return(version_);}
  timespec collectionTimeSummary() { return(collectionTimeSummary_.get_timespec());}

  double deltaT()       const { return(deltaT_);}
  double deltaTActive() const { return(deltaTActive_);}
  double deltaTRun()       const { return(deltaTRun_);}
  double deltaTRunActive() const { return(deltaTRunActive_);}

  // Instantaneous Rate accessors
  double triggerNumberRate() const            
  { return(triggerNumberRate_);}

  double eventNumberRate() const              
  { return(eventNumberRate_);}

  double finalTriggersGeneratedRate() const         
  { return(finalTriggersGeneratedRate_);}
  double finalTriggersDistributedRate() const         
  { return(finalTriggersDistributedRate_);}

  double randomTriggersRate() const          
  { return(randomTriggersRate_);}

  double calibrationTriggersRate() const     
  { return(calibrationTriggersRate_);}

  double totalTestTriggersRate() const                 
  { return(totalTestTriggersRate_);}

  double orbitNumberRate() const              
  { return(orbitNumberRate_);}

  double numberResetsRate() const             
  { return(numberResetsRate_);}

  double deadTimePercent() const                 
  { return(deadTimePercent_);}

  double deadTimeActivePercent() const           
  { return(deadTimeActivePercent_);}

  double deadTimeActiveCalibrationPercent() const
  { return(deadTimeActiveCalibrationPercent_);}

  double deadTimeActivePrivatePercent() const    
  { return(deadTimeActivePrivatePercent_);}

  double deadTimeActivePartitionPercent() const  
  { return(deadTimeActivePartitionPercent_);}

  double deadTimeActiveThrottlePercent() const   
  { return(deadTimeActiveThrottlePercent_);}

  double deadTimeActiveTimeSlotPercent() const   
  { return(deadTimeActiveTimeSlotPercent_);}

  double finalTriggersInvalidBCPercent() const       
  { return(finalTriggersInvalidBCPercent_);}

  double lostFinalTriggersPercent() const             
  { return(lostFinalTriggersPercent_);}

  double lostFinalTriggersActivePercent() const       
  { return(lostFinalTriggersActivePercent_);}

  timespec collectionTimeDetails() const 
  { return(collectionTimeDetails_.get_timespec());}

  std::vector<double> triggersRate() const    { return(triggersRate_);}
  std::vector<double> testTriggersRate() const
  { return(testTriggersRate_);}

  // Run Rate Accessors
  double triggerNumberRunRate() const            
  { return(triggerNumberRunRate_);}

  double eventNumberRunRate() const              
  { return(eventNumberRunRate_);}

  double finalTriggersDistributedRunRate() const         
  { return(finalTriggersDistributedRunRate_);}

  double finalTriggersGeneratedRunRate() const      
  { return(finalTriggersGeneratedRunRate_);}

  double randomTriggersRunRate() const          
  { return(randomTriggersRunRate_);}

  double calibrationTriggersRunRate() const     
  { return(calibrationTriggersRunRate_);}

  double totalTestTriggersRunRate() const                 
  { return(totalTestTriggersRunRate_);}

  double orbitNumberRunRate() const              
  { return(orbitNumberRunRate_);}

  double numberResetsRunRate() const             
  { return(numberResetsRunRate_);}

  double deadTimeRunPercent() const                 
  { return(deadTimeRunPercent_);}

  double deadTimeActiveRunPercent() const           
  { return(deadTimeActiveRunPercent_);}

  double deadTimeActiveCalibrationRunPercent() const
  { return(deadTimeActiveCalibrationRunPercent_);}

  double deadTimeActivePrivateRunPercent() const    
  { return(deadTimeActivePrivateRunPercent_);}

  double deadTimeActivePartitionRunPercent() const  
  { return(deadTimeActivePartitionRunPercent_);}

  double deadTimeActiveThrottleRunPercent() const   
  { return(deadTimeActiveThrottleRunPercent_);}

  double deadTimeActiveTimeSlotRunPercent() const   
  { return(deadTimeActiveTimeSlotRunPercent_);}

  double finalTriggersInvalidBCRunPercent() const       
  { return(finalTriggersInvalidBCRunPercent_);}

  double lostFinalTriggersRunPercent() const             
  { return(lostFinalTriggersRunPercent_);}

  double lostFinalTriggersActiveRunPercent() const       
  { return(lostFinalTriggersActiveRunPercent_);}

  std::vector<double> triggersRunRate() const    
  { return(triggersRunRate_);}

  std::vector<double> testTriggersRunRate() const    
  { return(testTriggersRunRate_);}

  /// equality operator
  int operator==(const L1TriggerRates& e) const { return false; }

  /// inequality operator
  int operator!=(const L1TriggerRates& e) const { return false; }

protected:

  int version_;
  TimeSpec collectionTimeSummary_;

  double deltaT_;
  double deltaTActive_;

  double triggerNumberRate_;
  double eventNumberRate_;
  double finalTriggersDistributedRate_;
  double finalTriggersGeneratedRate_;
  double randomTriggersRate_;
  double calibrationTriggersRate_;
  double totalTestTriggersRate_;
  double orbitNumberRate_;
  double numberResetsRate_;
  double deadTimePercent_;
  double deadTimeActivePercent_;
  double deadTimeActiveCalibrationPercent_;
  double deadTimeActivePrivatePercent_;
  double deadTimeActivePartitionPercent_;
  double deadTimeActiveThrottlePercent_;
  double deadTimeActiveTimeSlotPercent_;
  double finalTriggersInvalidBCPercent_;
  double lostFinalTriggersPercent_;
  double lostFinalTriggersActivePercent_;

  std::vector<double> triggersRate_;
  std::vector<double> testTriggersRate_;

  double deltaTRun_;
  double deltaTRunActive_;

  double triggerNumberRunRate_;
  double eventNumberRunRate_;
  double finalTriggersDistributedRunRate_;
  double finalTriggersGeneratedRunRate_;
  double randomTriggersRunRate_;
  double calibrationTriggersRunRate_;
  double totalTestTriggersRunRate_;
  double orbitNumberRunRate_;
  double numberResetsRunRate_;
  double deadTimeRunPercent_;
  double deadTimeActiveRunPercent_;
  double deadTimeActiveCalibrationRunPercent_;
  double deadTimeActivePrivateRunPercent_;
  double deadTimeActivePartitionRunPercent_;
  double deadTimeActiveThrottleRunPercent_;
  double deadTimeActiveTimeSlotRunPercent_;
  double finalTriggersInvalidBCRunPercent_;
  double lostFinalTriggersRunPercent_;
  double lostFinalTriggersActiveRunPercent_;

  TimeSpec collectionTimeDetails_;
  std::vector<double> triggersRunRate_;
  std::vector<double> testTriggersRunRate_;
};


/// Pretty-print operator for L1TriggerRates
std::ostream& operator<<(std::ostream& s, const L1TriggerRates& c);

typedef std::vector<L1TriggerRates> L1TriggerRatesCollection;

#endif
