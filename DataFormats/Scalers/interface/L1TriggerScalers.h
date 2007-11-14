/*
 *  File: DataFormats/Scalers/interface/L1TriggerScalers.h   (W.Badgett)
 *
 *  Various Level 1 Trigger Scalers from the GT/TS
 *
 */

#ifndef L1TRIGGERSCALERS_H
#define L1TRIGGERSCALERS_H

#include <ostream>


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
  L1TriggerScalers(uint16_t rawData);
  virtual ~L1TriggerScalers();

  /// name method
  std::string name() const { return "L1TriggerScalers"; }

  /// empty method (= false)
  bool empty() const { return false; }

  /// get the data
  uint16_t raw() const { return m_data; }

  struct timespec getCollectionTimeSummary() { return(collectionTimeSummary);}
  unsigned long getTriggerNumber() { return(triggerNumber);}
  unsigned long getEventNumber() { return(eventNumber);}
  unsigned long getPhysicsL1Accepts() { return(physicsL1Accepts);}
  unsigned long getPhysicsL1AcceptsRaw() { return(physicsL1AcceptsRaw);}
  unsigned long getRandomL1Accepts() { return(randomL1Accepts);}
  unsigned long getCalibrationL1Accepts() { return(calibrationL1Accepts);}
  unsigned long getTechTrig() { return(techTrig);}
  unsigned long getOrbitNumber() { return(orbitNumber);}
  unsigned long getNumberResets() { return(numberResets);}
  unsigned long getDeadTime() { return(deadTime);}
  unsigned long getDeadTimeActive() { return(deadTimeActive);}
  unsigned long getDeadTimeActiveCalibration() { return(deadTimeActiveCalibration);}
  unsigned long getDeadTimeActivePrivate() { return(deadTimeActivePrivate);}
  unsigned long getDeadTimeActivePartition() { return(deadTimeActivePartition);}
  unsigned long getDeadTimeActiveThrottle() { return(deadTimeActiveThrottle);}

  struct timespec getCollectionTimeDetails() { return(collectionTimeDetails);}
  unsigned long * getTriggers() { return(triggers);}

  /// equality operator
  int operator==(const L1TriggerScalers& e) const { return m_data==e.raw(); }

  /// inequality operator
  int operator!=(const L1TriggerScalers& e) const { return m_data!=e.raw(); }

protected:

  uint16_t m_data;
  int version;

  struct timespec collectionTimeSummary;
  unsigned long triggerNumber;
  unsigned long eventNumber;
  unsigned long physicsL1Accepts;
  unsigned long physicsL1AcceptsRaw;
  unsigned long randomL1Accepts;
  unsigned long calibrationL1Accepts;
  unsigned long techTrig;
  unsigned long orbitNumber;
  unsigned long numberResets;
  unsigned long deadTime;
  unsigned long deadTimeActive;
  unsigned long deadTimeActiveCalibration;
  unsigned long deadTimeActivePrivate;
  unsigned long deadTimeActivePartition;
  unsigned long deadTimeActiveThrottle;

  struct timespec collectionTimeDetails;
  unsigned long triggers[nL1Triggers];
};


/// Pretty-print operator for L1TriggerScalers
std::ostream& operator<<(std::ostream& s, const L1TriggerScalers& c);


#endif
