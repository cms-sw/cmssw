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
  unsigned long long triggerNumber() { return(triggerNumber_);}
  unsigned long long eventNumber() { return(eventNumber_);}
  unsigned long long physicsL1Accepts() { return(physicsL1Accepts_);}
  unsigned long long physicsL1AcceptsRaw() { return(physicsL1AcceptsRaw_);}
  unsigned long long randomL1Accepts() { return(randomL1Accepts_);}
  unsigned long long calibrationL1Accepts() { return(calibrationL1Accepts_);}
  unsigned long long techTrig() { return(techTrig_);}
  unsigned long long orbitNumber() { return(orbitNumber_);}
  unsigned long long numberResets() { return(numberResets_);}
  unsigned long long deadTime() { return(deadTime_);}
  unsigned long long deadTimeActive() { return(deadTimeActive_);}
  unsigned long long deadTimeActiveCalibration() { return(deadTimeActiveCalibration_);}
  unsigned long long deadTimeActivePrivate() { return(deadTimeActivePrivate_);}
  unsigned long long deadTimeActivePartition() { return(deadTimeActivePartition_);}
  unsigned long long deadTimeActiveThrottle() { return(deadTimeActiveThrottle_);}

  struct timespec CollectionTimeDetails() { return(collectionTimeDetails_);}
  std::vector<unsigned long long> Triggers() { return(triggers_);}

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
  unsigned long long techTrig_;
  unsigned long long orbitNumber_;
  unsigned long long numberResets_;
  unsigned long long deadTime_;
  unsigned long long deadTimeActive_;
  unsigned long long deadTimeActiveCalibration_;
  unsigned long long deadTimeActivePrivate_;
  unsigned long long deadTimeActivePartition_;
  unsigned long long deadTimeActiveThrottle_;

  struct timespec collectionTimeDetails_;
  std::vector<unsigned long long> triggers_;
};


/// Pretty-print operator for L1TriggerScalers
std::ostream& operator<<(std::ostream& s, const L1TriggerScalers& c);

typedef std::vector<L1TriggerScalers> L1TriggerScalersCollection;

#endif
