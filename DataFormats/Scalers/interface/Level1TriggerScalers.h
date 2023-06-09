/*
 *  File: DataFormats/Scalers/interface/Level1TriggerScalers.h   (W.Badgett)
 *
 *  Various Level 1 Trigger Scalers from the GT/TS
 *
 */

#ifndef DATAFORMATS_SCALERS_LEVEL1TRIGGERSCALERS_H
#define DATAFORMATS_SCALERS_LEVEL1TRIGGERSCALERS_H

#include <ostream>
#include <vector>

#include "DataFormats/Scalers/interface/TimeSpec.h"

/*! \file Level1TriggerScalers.h
 * \Header file for Level 1 Global Trigger Scalers
 * 
 * \author: William Badgett
 *
 */

/// \class Level1TriggerScalers.h
/// \brief Persistable copy of Level1 Trigger Scalers

class Level1TriggerScalers {
public:
  enum { nLevel1Triggers = 128, nLevel1TestTriggers = 64, firstShortLSRun = 125574 };

  static const unsigned long long N_BX = 3564ULL;
  static const unsigned long long N_BX_ACTIVE = 2808ULL;
  static const unsigned long long N_ORBITS_LUMI_SECTION = 0x100000ULL;
  static const unsigned long long N_BX_LUMI_SECTION = N_ORBITS_LUMI_SECTION * N_BX;

  Level1TriggerScalers();
  Level1TriggerScalers(const unsigned char* rawData);
  virtual ~Level1TriggerScalers();

  /// name method
  std::string name() const { return "Level1TriggerScalers"; }

  /// empty method (= false)
  bool empty() const { return false; }

  // Data accessor methods
  int version() const { return (version_); }

  unsigned int trigType() const { return (trigType_); }
  unsigned int eventID() const { return (eventID_); }
  unsigned int sourceID() const { return (sourceID_); }
  unsigned int bunchNumber() const { return (bunchNumber_); }

  struct timespec collectionTime() const { return (collectionTime_.get_timespec()); }

  unsigned int lumiSegmentNr() const { return (lumiSegmentNr_); }
  unsigned int lumiSegmentOrbits() const { return (lumiSegmentOrbits_); }
  unsigned int orbitNr() const { return (orbitNr_); }

  unsigned int gtResets() const { return (gtResets_); }
  unsigned int bunchCrossingErrors() const { return (bunchCrossingErrors_); }
  unsigned long long gtTriggers() const { return (gtTriggers_); }
  unsigned long long gtEvents() const { return (gtEvents_); }
  float gtTriggersRate() const { return (gtTriggersRate_); }
  float gtEventsRate() const { return (gtEventsRate_); }
  int prescaleIndexAlgo() const { return (prescaleIndexAlgo_); }
  int prescaleIndexTech() const { return (prescaleIndexTech_); }

  struct timespec collectionTimeLumiSeg() const { return (collectionTimeLumiSeg_.get_timespec()); }

  unsigned int lumiSegmentNrLumiSeg() const { return (lumiSegmentNrLumiSeg_); }

  unsigned long long triggersPhysicsGeneratedFDL() const { return (triggersPhysicsGeneratedFDL_); }
  unsigned long long triggersPhysicsLost() const { return (triggersPhysicsLost_); }
  unsigned long long triggersPhysicsLostBeamActive() const { return (triggersPhysicsLostBeamActive_); }
  unsigned long long triggersPhysicsLostBeamInactive() const { return (triggersPhysicsLostBeamInactive_); }
  unsigned long long l1AsPhysics() const { return (l1AsPhysics_); }
  unsigned long long l1AsRandom() const { return (l1AsRandom_); }
  unsigned long long l1AsTest() const { return (l1AsTest_); }
  unsigned long long l1AsCalibration() const { return (l1AsCalibration_); }
  unsigned long long deadtime() const { return (deadtime_); }
  unsigned long long deadtimeBeamActive() const { return (deadtimeBeamActive_); }
  unsigned long long deadtimeBeamActiveTriggerRules() const { return (deadtimeBeamActiveTriggerRules_); }
  unsigned long long deadtimeBeamActiveCalibration() const { return (deadtimeBeamActiveCalibration_); }
  unsigned long long deadtimeBeamActivePrivateOrbit() const { return (deadtimeBeamActivePrivateOrbit_); }
  unsigned long long deadtimeBeamActivePartitionController() const { return (deadtimeBeamActivePartitionController_); }
  unsigned long long deadtimeBeamActiveTimeSlot() const { return (deadtimeBeamActiveTimeSlot_); }

  unsigned int lastOrbitCounter0() const { return (lastOrbitCounter0_); }
  unsigned int lastTestEnable() const { return (lastTestEnable_); }
  unsigned int lastResync() const { return (lastResync_); }
  unsigned int lastStart() const { return (lastStart_); }
  unsigned int lastEventCounter0() const { return (lastEventCounter0_); }
  unsigned int lastHardReset() const { return (lastHardReset_); }
  unsigned long long spare0() const { return (spare0_); }
  unsigned long long spare1() const { return (spare1_); }
  unsigned long long spare2() const { return (spare2_); }

  static double rateLS(unsigned long long counts);
  static double rateLS(unsigned int counts);
  static double percentLS(unsigned long long counts);
  static double percentLSActive(unsigned long long counts);

  static double rateLS(unsigned long long counts, int runNumber);
  static double rateLS(unsigned int counts, int runNumber);
  static double percentLS(unsigned long long counts, int runNumber);
  static double percentLSActive(unsigned long long counts, int runNumber);

  std::vector<unsigned int> gtAlgoCounts() const { return (gtAlgoCounts_); }

  std::vector<unsigned int> gtTechCounts() const { return (gtTechCounts_); }

  /// equality operator
  int operator==(const Level1TriggerScalers& e) const { return false; }

  /// inequality operator
  int operator!=(const Level1TriggerScalers& e) const { return false; }

protected:
  int version_;

  unsigned int trigType_;
  unsigned int eventID_;
  unsigned int sourceID_;
  unsigned int bunchNumber_;

  TimeSpec collectionTime_;
  unsigned int lumiSegmentNr_;
  unsigned int lumiSegmentOrbits_;
  unsigned int orbitNr_;
  unsigned int gtResets_;
  unsigned int bunchCrossingErrors_;
  unsigned long long gtTriggers_;
  unsigned long long gtEvents_;
  float gtTriggersRate_;
  float gtEventsRate_;
  int prescaleIndexAlgo_;
  int prescaleIndexTech_;

  TimeSpec collectionTimeLumiSeg_;
  unsigned int lumiSegmentNrLumiSeg_;
  unsigned long long triggersPhysicsGeneratedFDL_;
  unsigned long long triggersPhysicsLost_;
  unsigned long long triggersPhysicsLostBeamActive_;
  unsigned long long triggersPhysicsLostBeamInactive_;
  unsigned long long l1AsPhysics_;
  unsigned long long l1AsRandom_;
  unsigned long long l1AsTest_;
  unsigned long long l1AsCalibration_;
  unsigned long long deadtime_;
  unsigned long long deadtimeBeamActive_;
  unsigned long long deadtimeBeamActiveTriggerRules_;
  unsigned long long deadtimeBeamActiveCalibration_;
  unsigned long long deadtimeBeamActivePrivateOrbit_;
  unsigned long long deadtimeBeamActivePartitionController_;
  unsigned long long deadtimeBeamActiveTimeSlot_;

  std::vector<unsigned int> gtAlgoCounts_;
  std::vector<unsigned int> gtTechCounts_;

  // Orbit counter markers indicating when the last BGO
  // command of a particular type was received, relative
  // to the last OrbitCounter0 (OC0), for this L1 accept
  unsigned int lastOrbitCounter0_;
  unsigned int lastTestEnable_;
  unsigned int lastResync_;
  unsigned int lastStart_;
  unsigned int lastEventCounter0_;
  unsigned int lastHardReset_;

  // For future use
  unsigned long long spare0_;
  unsigned long long spare1_;
  unsigned long long spare2_;
};

/// Pretty-print operator for Level1TriggerScalers
std::ostream& operator<<(std::ostream& s, const Level1TriggerScalers& c);

typedef std::vector<Level1TriggerScalers> Level1TriggerScalersCollection;

#endif
