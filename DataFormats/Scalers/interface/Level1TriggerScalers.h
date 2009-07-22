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

/*! \file Level1TriggerScalers.h
 * \Header file for Level 1 Global Trigger Scalers
 * 
 * \author: William Badgett
 *
 */


/// \class Level1TriggerScalers.h
/// \brief Persistable copy of Level1 Trigger Scalers

class Level1TriggerScalers
{
 public:

  enum 
  {
    nLevel1Triggers          = 128,
    nLevel1TestTriggers      = 64
  };

  Level1TriggerScalers();
  Level1TriggerScalers(const unsigned char * rawData);
  virtual ~Level1TriggerScalers();

  /// name method
  std::string name() const { return "Level1TriggerScalers"; }

  /// empty method (= false)
  bool empty() const { return false; }

  // Data accessor methods
  int version() const { return(version_);}

  unsigned int trigType() const            { return(trigType_);}
  unsigned int eventID() const             { return(eventID_);}
  unsigned int sourceID() const            { return(sourceID_);}
  unsigned int bunchNumber() const         { return(bunchNumber_);}

  struct timespec collectionTimeGeneral() const
  { return(collectionTimeGeneral_);}

  unsigned int lumiSegmentNr() const        { return(lumiSegmentNr_);}
  unsigned int lumiSegmentOrbits() const    { return(lumiSegmentOrbits_);}
  unsigned int orbitNr() const              { return(orbitNr_);}

  unsigned int gtPartition0Resets() const   { return(gtPartition0Resets_);}
  unsigned int bunchCrossingErrors() const  { return(bunchCrossingErrors_);}
  unsigned int gtPartition0Triggers() const { return(gtPartition0Triggers_);}
  unsigned int gtPartition0Events() const   { return(gtPartition0Events_);}
  int prescaleIndexAlgo() const             { return(prescaleIndexAlgo_);}
  int prescaleIndexTech() const             { return(prescaleIndexTech_);}

  struct timespec collectionTimeLumiSeg() const 
  { return(collectionTimeLumiSeg_);}

  unsigned int lumiSegmentNrLumiSeg() const      
  { return(lumiSegmentNrLumiSeg_);}

  unsigned long long () const  
  { return(_);}

  unsigned long long triggersPhysicsGeneratedFDL() const 
  { return(triggersPhysicsGeneratedFDL_);}
  unsigned long long triggersPhysicsLost() const 
  { return(triggersPhysicsLost_);}
  unsigned long long triggersPhysicsLostBeamActive() const 
  { return(triggersPhysicsLostBeamActive_);}
  unsigned long long triggersPhysicsLostBeamInactive() const 
  { return(triggersPhysicsLostBeamInactive_);}
  unsigned long long l1AsPhysics() const 
  { return(l1AsPhysics_);}
  unsigned long long l1AsRandom() const 
  { return(l1AsRandom_);}
  unsigned long long l1AsTest() const 
  { return(l1AsTest_);}
  unsigned long long l1AsCalibration() const 
  { return(l1AsCalibration_);}
  unsigned long long deadtime() const 
  { return(deadtime_);}
  unsigned long long deadtimeBeamActive() const 
  { return(deadtimeBeamActive_);}
  unsigned long long deadtimeBeamActiveTriggerRules() const 
  { return(deadtimeBeamActiveTriggerRules_);}
  unsigned long long deadtimeBeamActiveCalibration() const 
  { return(deadtimeBeamActiveCalibration_);}
  unsigned long long deadtimeBeamActivePrivateOrbit() const 
  { return(deadtimeBeamActivePrivateOrbit_);}
  unsigned long long deadtimeBeamActivePartitionController() const 
  { return(deadtimeBeamActivePartitionController_);}
  unsigned long long deadtimeBeamActiveTimeSlot() const 
  { return(deadtimeBeamActiveTimeSlot_);}

  std::vector<unsigned int> gtAlgoCounts() const 
  { return(gtAlgoCounts_);}

  std::vector<unsigned int> gtTechCounts() const
  { return(gtTechCounts_);}

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

  struct timespec    collectionTimeGeneral_;
  unsigned int lumiSegmentNr_;
  unsigned int lumiSegmentOrbits_;
  unsigned int orbitNr_;
  unsigned int gtPartition0Resets_;
  unsigned int bunchCrossingErrors_;
  unsigned int gtPartition0Triggers_;
  unsigned int gtPartition0Events_;
  int prescaleIndexAlgo_;
  int prescaleIndexTech_;

  struct timespec    collectionTimeLumiSeg_;
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
};


/// Pretty-print operator for Level1TriggerScalers
std::ostream& operator<<(std::ostream& s, const Level1TriggerScalers& c);

typedef std::vector<Level1TriggerScalers> Level1TriggerScalersCollection;

#endif
