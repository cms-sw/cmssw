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

  struct timespec collectionTimeSummary() { return(collectionTimeSummary_);}

  double deltaT() const { return(deltaT_);}

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

  double techTrigRate() const                 
  { return(techTrigRate_);}

  double orbitNumberRate() const              
  { return(orbitNumberRate_);}

  double numberResetsRate() const             
  { return(numberResetsRate_);}

  double deadTimeRate() const                 
  { return(deadTimeRate_);}

  double deadTimeActiveRate() const           
  { return(deadTimeActiveRate_);}

  double deadTimeActiveCalibrationRate() const
  
  { return(deadTimeActiveCalibrationRate_);}

  double deadTimeActivePrivateRate() const    
  { return(deadTimeActivePrivateRate_);}

  double deadTimeActivePartitionRate() const  
  { return(deadTimeActivePartitionRate_);}

  double deadTimeActiveThrottleRate() const   
  { return(deadTimeActiveThrottleRate_);}

  double lostBunchCrossingsRate() const       
  { return(lostBunchCrossingsRate_);}

  double lostTriggersRate() const             
  { return(lostTriggersRate_);}

  double lostTriggersActiveRate() const       
  { return(lostTriggersActiveRate_);}

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

  double techTrigRunRate() const                 
  { return(techTrigRunRate_);}

  double orbitNumberRunRate() const              
  { return(orbitNumberRunRate_);}

  double numberResetsRunRate() const             
  { return(numberResetsRunRate_);}

  double deadTimeRunRate() const                 
  { return(deadTimeRunRate_);}

  double deadTimeActiveRunRate() const           
  { return(deadTimeActiveRunRate_);}

  double deadTimeActiveCalibrationRunRate() const
  
  { return(deadTimeActiveCalibrationRunRate_);}

  double deadTimeActivePrivateRunRate() const    
  { return(deadTimeActivePrivateRunRate_);}

  double deadTimeActivePartitionRunRate() const  
  { return(deadTimeActivePartitionRunRate_);}

  double deadTimeActiveThrottleRunRate() const   
  { return(deadTimeActiveThrottleRunRate_);}

  double lostBunchCrossingsRunRate() const       
  { return(lostBunchCrossingsRunRate_);}

  double lostTriggersRunRate() const             
  { return(lostTriggersRunRate_);}

  double lostTriggersActiveRunRate() const       
  { return(lostTriggersActiveRunRate_);}

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

  double triggerNumberRate_;
  double eventNumberRate_;
  double physicsL1AcceptsRate_;
  double physicsL1AcceptsRawRate_;
  double randomL1AcceptsRate_;
  double calibrationL1AcceptsRate_;
  double techTrigRate_;
  double orbitNumberRate_;
  double numberResetsRate_;
  double deadTimeRate_;
  double deadTimeActiveRate_;
  double deadTimeActiveCalibrationRate_;
  double deadTimeActivePrivateRate_;
  double deadTimeActivePartitionRate_;
  double deadTimeActiveThrottleRate_;
  double lostBunchCrossingsRate_;
  double lostTriggersRate_;
  double lostTriggersActiveRate_;

  std::vector<double> triggersRate_;

  double triggerNumberRunRate_;
  double eventNumberRunRate_;
  double physicsL1AcceptsRunRate_;
  double physicsL1AcceptsRawRunRate_;
  double randomL1AcceptsRunRate_;
  double calibrationL1AcceptsRunRate_;
  double techTrigRunRate_;
  double orbitNumberRunRate_;
  double numberResetsRunRate_;
  double deadTimeRunRate_;
  double deadTimeActiveRunRate_;
  double deadTimeActiveCalibrationRunRate_;
  double deadTimeActivePrivateRunRate_;
  double deadTimeActivePartitionRunRate_;
  double deadTimeActiveThrottleRunRate_;
  double lostBunchCrossingsRunRate_;
  double lostTriggersRunRate_;
  double lostTriggersActiveRunRate_;

  struct timespec collectionTimeDetails_;
  std::vector<double> triggersRunRate_;
};


/// Pretty-print operator for L1TriggerRates
std::ostream& operator<<(std::ostream& s, const L1TriggerRates& c);

typedef std::vector<L1TriggerRates> L1TriggerRatesCollection;

#endif
