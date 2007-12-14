/*
 *  File: DataFormats/Scalers/interface/L1TriggerRates.h   (W.Badgett)
 *
 *  Various Level 1 Trigger Rates from the GT/TS
 *
 */

#ifndef L1TRIGGERRATES_H
#define L1TRIGGERRATES_H

#include <ostream>
#include <vector>

/*! \file L1TriggerRates.h
 * \Header file for Level 1 Global Trigger Rates
 * 
 * \author: William Badgett
 *
 */


/// \class L1TriggerRates.h
/// \brief Persistable copy of L1 Trigger Rates

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
  L1TriggerRates(const L1TriggerScalers s);
  L1TriggerRates(const L1TriggerScalers s1, const L1TriggerScalers s2);
  virtual ~L1TriggerRates();

  void computeRunRates(const L1TriggerScalers t);
  void computeRates(const L1TriggerScalers t1,
		    const L1TriggerScalers t2);

  /// name method
  std::string name() const { return "L1TriggerRates"; }

  /// empty method (= false)
  bool empty() const { return false; }

  /// get the data

  int version() const { return(version_);}
  struct timespec collectionTimeSummary() { return(collectionTimeSummary_);}

  double deltaT()       const { return(deltaT_);}
  double deltaTActive() const { return(deltaTActive_);}
  double deltaTRun()       const { return(deltaTRun_);}
  double deltaTRunActive() const { return(deltaTRunActive_);}

  // Instantaneous Rate accessors
  double triggerNumberRate() const            
  { return(triggerNumberRate_);}

  double eventNumberRate() const              
  { return(eventNumberRate_);}

  double physicsL1AcceptsRate() const         
  { return(physicsL1AcceptsRate_);}

  double physicsL1AcceptsRawRate() const      
  { return(physicsL1AcceptsRawRate_);}

  double randomL1AcceptsRate() const          
  { return(randomL1AcceptsRate_);}

  double calibrationL1AcceptsRate() const     
  { return(calibrationL1AcceptsRate_);}

  double technicalTriggersRate() const                 
  { return(technicalTriggersRate_);}

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

  double lostBunchCrossingsPercent() const       
  { return(lostBunchCrossingsPercent_);}

  double lostTriggersPercent() const             
  { return(lostTriggersPercent_);}

  double lostTriggersActivePercent() const       
  { return(lostTriggersActivePercent_);}

  struct timespec collectionTimeDetails() const 
  { return(collectionTimeDetails_);}

  std::vector<double> triggersRate() const    { return(triggersRate_);}

  // Run Rate Accessors
  double triggerNumberRunRate() const            
  { return(triggerNumberRunRate_);}

  double eventNumberRunRate() const              
  { return(eventNumberRunRate_);}

  double physicsL1AcceptsRunRate() const         
  { return(physicsL1AcceptsRunRate_);}

  double physicsL1AcceptsRawRunRate() const      
  { return(physicsL1AcceptsRawRunRate_);}

  double randomL1AcceptsRunRate() const          
  { return(randomL1AcceptsRunRate_);}

  double calibrationL1AcceptsRunRate() const     
  { return(calibrationL1AcceptsRunRate_);}

  double technicalTriggersRunRate() const                 
  { return(technicalTriggersRunRate_);}

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

  double lostBunchCrossingsRunPercent() const       
  { return(lostBunchCrossingsRunPercent_);}

  double lostTriggersRunPercent() const             
  { return(lostTriggersRunPercent_);}

  double lostTriggersActiveRunPercent() const       
  { return(lostTriggersActiveRunPercent_);}

  std::vector<double> triggersRunRate() const    
  { return(triggersRunRate_);}

  /// equality operator
  int operator==(const L1TriggerRates& e) const { return false; }

  /// inequality operator
  int operator!=(const L1TriggerRates& e) const { return false; }

protected:

  int version_;
  struct timespec collectionTimeSummary_;

  double deltaT_;
  double deltaTActive_;

  double triggerNumberRate_;
  double eventNumberRate_;
  double physicsL1AcceptsRate_;
  double physicsL1AcceptsRawRate_;
  double randomL1AcceptsRate_;
  double calibrationL1AcceptsRate_;
  double technicalTriggersRate_;
  double orbitNumberRate_;
  double numberResetsRate_;
  double deadTimePercent_;
  double deadTimeActivePercent_;
  double deadTimeActiveCalibrationPercent_;
  double deadTimeActivePrivatePercent_;
  double deadTimeActivePartitionPercent_;
  double deadTimeActiveThrottlePercent_;
  double lostBunchCrossingsPercent_;
  double lostTriggersPercent_;
  double lostTriggersActivePercent_;

  std::vector<double> triggersRate_;

  double deltaTRun_;
  double deltaTRunActive_;

  double triggerNumberRunRate_;
  double eventNumberRunRate_;
  double physicsL1AcceptsRunRate_;
  double physicsL1AcceptsRawRunRate_;
  double randomL1AcceptsRunRate_;
  double calibrationL1AcceptsRunRate_;
  double technicalTriggersRunRate_;
  double orbitNumberRunRate_;
  double numberResetsRunRate_;
  double deadTimeRunPercent_;
  double deadTimeActiveRunPercent_;
  double deadTimeActiveCalibrationRunPercent_;
  double deadTimeActivePrivateRunPercent_;
  double deadTimeActivePartitionRunPercent_;
  double deadTimeActiveThrottleRunPercent_;
  double lostBunchCrossingsRunPercent_;
  double lostTriggersRunPercent_;
  double lostTriggersActiveRunPercent_;

  struct timespec collectionTimeDetails_;
  std::vector<double> triggersRunRate_;
};


/// Pretty-print operator for L1TriggerRates
std::ostream& operator<<(std::ostream& s, const L1TriggerRates& c);

typedef std::vector<L1TriggerRates> L1TriggerRatesCollection;

#endif
