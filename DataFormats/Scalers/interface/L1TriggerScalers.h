/*
 *  File: DataFormats/Scalers/interface/L1TriggerScalers.h   (W.Badgett)
 *
 *  Various Level 1 Trigger Scalers from the GT/TS
 *
 */

#ifndef L1TRIGGERSCALERS_H
#define L1TRIGGERSCALERS_H

#include <ostream>
#include <vector>

/*! \file L1TriggerScalers.h
 * \Header file for Level 1 Global Trigger Scalers
 * 
 * \author: William Badgett
 *
 */


/// \class L1TriggerScalers.h
/// \brief Persistable copy of L1 Trigger Scalers

class L1TriggerScalers
{
 public:

  enum 
  {
    nL1Triggers      = 128
  };

  L1TriggerScalers();
  L1TriggerScalers(const unsigned char * rawData);
  virtual ~L1TriggerScalers();

  /// name method
  std::string name() const { return "L1TriggerScalers"; }

  /// empty method (= false)
  bool empty() const { return false; }

  /// get the data

  int version() const { return(version_);}

  struct timespec collectionTimeSummary() const 
  { return(collectionTimeSummary_);}

  unsigned long long triggerNumber() const         
  { return(triggerNumber_);}

  unsigned long long eventNumber() const           
  { return(eventNumber_);}

  unsigned long long physicsL1Accepts() const      
  { return(physicsL1Accepts_);}

  unsigned long long physicsL1AcceptsRaw() const   
  { return(physicsL1AcceptsRaw_);}

  unsigned long long randomL1Accepts() const       
  { return(randomL1Accepts_);}

  unsigned long long calibrationL1Accepts() const  
  { return(calibrationL1Accepts_);}

  unsigned long long technicalTriggers() const 
  { return(technicalTriggers_);}

  unsigned long long orbitNumber() const           
  { return(orbitNumber_);}

  unsigned long long numberResets() const          
  { return(numberResets_);}

  unsigned long long deadTime() const              
  { return(deadTime_);}

  unsigned long long deadTimeActive() const        
  { return(deadTimeActive_);}

  unsigned long long deadTimeActiveCalibration() const
  { return(deadTimeActiveCalibration_);}
  unsigned long long deadTimeActivePrivate() const   
  { return(deadTimeActivePrivate_);}
  unsigned long long deadTimeActivePartition() const 
  { return(deadTimeActivePartition_);}
  unsigned long long deadTimeActiveThrottle() const
  { return(deadTimeActiveThrottle_);}
  unsigned long long lostBunchCrossings() const
  { return(lostBunchCrossings_);}

  unsigned long long lostTriggers() const
  { return(lostTriggers_);}

  unsigned long long lostTriggersActive() const
  { return(lostTriggersActive_);}

  struct timespec collectionTimeDetails() const
  { return(collectionTimeDetails_);}
  std::vector<unsigned int> triggers() const   { return(triggers_);}

  /// equality operator
  int operator==(const L1TriggerScalers& e) const { return false; }

  /// inequality operator
  int operator!=(const L1TriggerScalers& e) const { return false; }

protected:

  int version_;
  struct timespec collectionTimeSummary_;
  unsigned long long triggerNumber_;
  unsigned long long eventNumber_;
  unsigned long long physicsL1Accepts_;
  unsigned long long physicsL1AcceptsRaw_;
  unsigned long long randomL1Accepts_;
  unsigned long long calibrationL1Accepts_;
  unsigned long long technicalTriggers_;
  unsigned long long orbitNumber_;
  unsigned long long numberResets_;
  unsigned long long deadTime_;
  unsigned long long deadTimeActive_;
  unsigned long long deadTimeActiveCalibration_;
  unsigned long long deadTimeActivePrivate_;
  unsigned long long deadTimeActivePartition_;
  unsigned long long deadTimeActiveThrottle_;
  unsigned long long lostBunchCrossings_;
  unsigned long long lostTriggers_;
  unsigned long long lostTriggersActive_;

  struct timespec collectionTimeDetails_;
  std::vector<unsigned int> triggers_;
};


/// Pretty-print operator for L1TriggerScalers
std::ostream& operator<<(std::ostream& s, const L1TriggerScalers& c);

typedef std::vector<L1TriggerScalers> L1TriggerScalersCollection;

#endif
