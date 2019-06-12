#ifndef DATAFORMATS_HCALRECHIT_HCALSOURCEPOSITIONDATA_H
#define DATAFORMATS_HCALRECHIT_HCALSOURCEPOSITIONDATA_H 1

#include <string>

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

class HcalSourcePositionData {
public:
  HcalSourcePositionData();
  ~HcalSourcePositionData() {}

  inline int messageCounter() const { return messageCounter_; }
  inline int status() const { return status_; }
  inline int indexCounter() const { return indexCounter_; }
  inline int reelCounter() const { return reelCounter_; }
  inline int motorCurrent() const { return motorCurrent_; }
  inline int speed() const { return -1; }  // no longer implemented
  inline int motorVoltage() const { return motorVoltage_; }
  inline int tubeId() const { return -1; }  // no longer implemented
  inline int driverId() const { return driverId_; }
  inline int sourceId() const { return sourceId_; }
  inline std::string tubeNameFromCoord() const { return tubeNameFromCoord_; }
  inline std::string tubeDescriptionFromSD() const { return tubeDescriptionFromSD_; }
  inline std::string lastCommand() const { return lastCommand_; }
  inline std::string message() const { return message_; }

  void getDriverTimestamp(int& seconds, int& useconds) const;
  void getDAQTimestamp(int& seconds, int& useconds) const;

  void set(int message_counter,
           int timestamp1_sec,
           int timestamp1_usec,
           int timestamp2_sec,
           int timestamp2_usec,
           int status,
           int index_counter,
           int reel_counter,
           int motor_current,
           int motor_voltage,
           int driver_id,
           int source_id,
           std::string tubeNameFromCoord,
           std::string tubeDescFromSD,
           std::string lastCommand,
           std::string message);

private:
  int messageCounter_;
  int indexCounter_;
  int reelCounter_;
  int timestamp1_sec_;
  int timestamp1_usec_;
  int timestamp2_sec_;
  int timestamp2_usec_;
  int status_;
  int motorCurrent_;
  int motorVoltage_;
  int tubeId_;
  int driverId_;
  int sourceId_;
  std::string tubeNameFromCoord_;
  std::string tubeDescriptionFromSD_;
  std::string lastCommand_;
  std::string message_;
};

std::ostream& operator<<(std::ostream& s, const HcalSourcePositionData& hspd);

#endif
