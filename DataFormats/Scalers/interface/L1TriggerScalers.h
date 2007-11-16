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

  struct timespec collectionTimeSummary() { return(collectionTimeSummary_);}
  unsigned int triggerNumber()            { return(triggerNumber_);}
  unsigned int eventNumber()              { return(eventNumber_);}
  unsigned int physicsL1Accepts()         { return(physicsL1Accepts_);}
  unsigned int physicsL1AcceptsRaw()      { return(physicsL1AcceptsRaw_);}
  unsigned int randomL1Accepts()          { return(randomL1Accepts_);}
  unsigned int calibrationL1Accepts()     { return(calibrationL1Accepts_);}
  unsigned int techTrig()                 { return(techTrig_);}
  unsigned int orbitNumber()              { return(orbitNumber_);}
  unsigned int numberResets()             { return(numberResets_);}
  unsigned int deadTime()                 { return(deadTime_);}
  unsigned int deadTimeActive()           { return(deadTimeActive_);}
  unsigned int deadTimeActiveCalibration()
  { return(deadTimeActiveCalibration_);}
  unsigned int deadTimeActivePrivate()    { return(deadTimeActivePrivate_);}
  unsigned int deadTimeActivePartition()  { return(deadTimeActivePartition_);}
  unsigned int deadTimeActiveThrottle()   { return(deadTimeActiveThrottle_);}
  unsigned int lostBunchCrossings()       { return(lostBunchCrossings_);}
  unsigned int lostTriggers()             { return(lostTriggers_);}
  unsigned int lostTriggersActive()       { return(lostTriggersActive_);}

  struct timespec CollectionTimeDetails() { return(collectionTimeDetails_);}
  std::vector<unsigned int> Triggers()    { return(triggers_);}

  /// equality operator
  int operator==(const L1TriggerScalers& e) const { return false; }

  /// inequality operator
  int operator!=(const L1TriggerScalers& e) const { return false; }

protected:

  int version_;
  struct timespec collectionTimeSummary_;
  unsigned int triggerNumber_;
  unsigned int eventNumber_;
  unsigned int physicsL1Accepts_;
  unsigned int physicsL1AcceptsRaw_;
  unsigned int randomL1Accepts_;
  unsigned int calibrationL1Accepts_;
  unsigned int techTrig_;
  unsigned int orbitNumber_;
  unsigned int numberResets_;
  unsigned int deadTime_;
  unsigned int deadTimeActive_;
  unsigned int deadTimeActiveCalibration_;
  unsigned int deadTimeActivePrivate_;
  unsigned int deadTimeActivePartition_;
  unsigned int deadTimeActiveThrottle_;
  unsigned int lostBunchCrossings_;
  unsigned int lostTriggers_;
  unsigned int lostTriggersActive_;

  struct timespec collectionTimeDetails_;
  std::vector<unsigned int> triggers_;
};


/// Pretty-print operator for L1TriggerScalers
std::ostream& operator<<(std::ostream& s, const L1TriggerScalers& c);

typedef std::vector<L1TriggerScalers> L1TriggerScalersCollection;

#endif
