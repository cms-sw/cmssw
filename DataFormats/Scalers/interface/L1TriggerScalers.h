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
  L1TriggerScalers(uint16_t rawData);
  virtual ~L1TriggerScalers();

  /// name method
  std::string name() const { return "L1TriggerScalers"; }

  /// empty method (= false)
  bool empty() const { return false; }

  /// get the data
  uint16_t raw() const { return m_data; }

  struct timespec getCollectionTimeSummary() { return(collectionTimeSummary);}
  unsigned long long getTriggerNumber() { return(triggerNumber);}
  unsigned long long getEventNumber() { return(eventNumber);}
  unsigned long long getPhysicsL1Accepts() { return(physicsL1Accepts);}
  unsigned long long getPhysicsL1AcceptsRaw() { return(physicsL1AcceptsRaw);}
  unsigned long long getRandomL1Accepts() { return(randomL1Accepts);}
  unsigned long long getCalibrationL1Accepts() { return(calibrationL1Accepts);}
  unsigned long long getTechTrig() { return(techTrig);}
  unsigned long long getOrbitNumber() { return(orbitNumber);}
  unsigned long long getNumberResets() { return(numberResets);}
  unsigned long long getDeadTime() { return(deadTime);}
  unsigned long long getDeadTimeActive() { return(deadTimeActive);}
  unsigned long long getDeadTimeActiveCalibration() { return(deadTimeActiveCalibration);}
  unsigned long long getDeadTimeActivePrivate() { return(deadTimeActivePrivate);}
  unsigned long long getDeadTimeActivePartition() { return(deadTimeActivePartition);}
  unsigned long long getDeadTimeActiveThrottle() { return(deadTimeActiveThrottle);}

  struct timespec getCollectionTimeDetails() { return(collectionTimeDetails);}
  std::vector<unsigned long long> getTriggers() { return(triggers);}

  /// equality operator
  int operator==(const L1TriggerScalers& e) const { return m_data==e.raw(); }

  /// inequality operator
  int operator!=(const L1TriggerScalers& e) const { return m_data!=e.raw(); }

protected:

  uint16_t m_data;
  int version;

  struct timespec collectionTimeSummary;
  unsigned long long triggerNumber;
  unsigned long long eventNumber;
  unsigned long long physicsL1Accepts;
  unsigned long long physicsL1AcceptsRaw;
  unsigned long long randomL1Accepts;
  unsigned long long calibrationL1Accepts;
  unsigned long long techTrig;
  unsigned long long orbitNumber;
  unsigned long long numberResets;
  unsigned long long deadTime;
  unsigned long long deadTimeActive;
  unsigned long long deadTimeActiveCalibration;
  unsigned long long deadTimeActivePrivate;
  unsigned long long deadTimeActivePartition;
  unsigned long long deadTimeActiveThrottle;

  struct timespec collectionTimeDetails;
  std::vector<unsigned long long> triggers;
};


/// Pretty-print operator for L1TriggerScalers
std::ostream& operator<<(std::ostream& s, const L1TriggerScalers& c);


#endif
