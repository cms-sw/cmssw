/*
 *  File: DataFormats/Scalers/interface/L1TriggerScalers.h   (W.Badgett)
 *
 *  Various Level 1 Trigger Scalers from the GT/TS
 *
 */

#ifndef DATAFORMATS_SCALERS_L1TRIGGERSCALERS_H
#define DATAFORMATS_SCALERS_L1TRIGGERSCALERS_H

#include "DataFormats/Scalers/interface/TimeSpec.h"

#include <ctime>
#include <iosfwd>
#include <string>
#include <vector>

/*! \file L1TriggerScalers.h
 * \Header file for Level 1 Global Trigger Scalers
 * 
 * \author: William Badgett
 *
 */

/// \class L1TriggerScalers.h
/// \brief Persistable copy of L1 Trigger Scalers

class L1TriggerScalers {
public:
  enum { nL1Triggers = 128, nL1TestTriggers = 64 };

  L1TriggerScalers();
  L1TriggerScalers(const unsigned char* rawData);
  virtual ~L1TriggerScalers();

  /// name method
  std::string name() const { return "L1TriggerScalers"; }

  /// empty method (= false)
  bool empty() const { return false; }

  // Data accessor methods
  int version() const { return (version_); }

  unsigned int trigType() const { return (trigType_); }
  unsigned int eventID() const { return (eventID_); }
  unsigned int sourceID() const { return (sourceID_); }
  unsigned int bunchNumber() const { return (bunchNumber_); }

  timespec collectionTimeSpecial() const { return (collectionTimeSpecial_.get_timespec()); }

  unsigned int orbitNumber() const { return (orbitNumber_); }
  unsigned int luminositySection() const { return (luminositySection_); }
  unsigned int bunchCrossingErrors() const { return (bunchCrossingErrors_); }

  timespec collectionTimeSummary() const { return (collectionTimeSummary_.get_timespec()); }

  unsigned int triggerNumber() const { return (triggerNumber_); }
  unsigned int eventNumber() const { return (eventNumber_); }
  unsigned int finalTriggersDistributed() const { return (finalTriggersDistributed_); }
  unsigned int calibrationTriggers() const { return (calibrationTriggers_); }
  unsigned int randomTriggers() const { return (randomTriggers_); }
  unsigned int totalTestTriggers() const { return (totalTestTriggers_); }
  unsigned int finalTriggersGenerated() const { return (finalTriggersGenerated_); }
  unsigned int finalTriggersInvalidBC() const { return (finalTriggersInvalidBC_); }

  unsigned long long deadTime() const { return (deadTime_); }
  unsigned long long lostFinalTriggers() const { return (lostFinalTriggers_); }
  unsigned long long deadTimeActive() const { return (deadTimeActive_); }
  unsigned long long lostFinalTriggersActive() const { return (lostFinalTriggersActive_); }

  unsigned long long deadTimeActivePrivate() const { return (deadTimeActivePrivate_); }
  unsigned long long deadTimeActivePartition() const { return (deadTimeActivePartition_); }
  unsigned long long deadTimeActiveThrottle() const { return (deadTimeActiveThrottle_); }
  unsigned long long deadTimeActiveCalibration() const { return (deadTimeActiveCalibration_); }
  unsigned long long deadTimeActiveTimeSlot() const { return (deadTimeActiveTimeSlot_); }
  unsigned int numberResets() const { return (numberResets_); }

  timespec collectionTimeDetails() const { return (collectionTimeDetails_.get_timespec()); }

  std::vector<unsigned int> triggers() const { return (triggers_); }

  std::vector<unsigned int> testTriggers() const { return (testTriggers_); }

  /// equality operator
  int operator==(const L1TriggerScalers& e) const { return false; }

  /// inequality operator
  int operator!=(const L1TriggerScalers& e) const { return false; }

protected:
  int version_;

  unsigned int trigType_;
  unsigned int eventID_;
  unsigned int sourceID_;
  unsigned int bunchNumber_;

  TimeSpec collectionTimeSpecial_;
  unsigned int orbitNumber_;
  unsigned int luminositySection_;
  unsigned short bunchCrossingErrors_;

  TimeSpec collectionTimeSummary_;
  unsigned int triggerNumber_;
  unsigned int eventNumber_;
  unsigned int finalTriggersDistributed_;
  unsigned int calibrationTriggers_;
  unsigned int randomTriggers_;
  unsigned int totalTestTriggers_;
  unsigned int finalTriggersGenerated_;
  unsigned int finalTriggersInvalidBC_;
  unsigned long long deadTime_;
  unsigned long long lostFinalTriggers_;
  unsigned long long deadTimeActive_;
  unsigned long long lostFinalTriggersActive_;
  unsigned long long deadTimeActivePrivate_;
  unsigned long long deadTimeActivePartition_;
  unsigned long long deadTimeActiveThrottle_;
  unsigned long long deadTimeActiveCalibration_;
  unsigned long long deadTimeActiveTimeSlot_;
  unsigned int numberResets_;

  TimeSpec collectionTimeDetails_;
  std::vector<unsigned int> triggers_;
  std::vector<unsigned int> testTriggers_;
};

/// Pretty-print operator for L1TriggerScalers
std::ostream& operator<<(std::ostream& s, const L1TriggerScalers& c);

typedef std::vector<L1TriggerScalers> L1TriggerScalersCollection;

#endif
