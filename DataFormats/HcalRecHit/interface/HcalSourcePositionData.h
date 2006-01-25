#ifndef DATAFORMATS_HCALRECHIT_HCALSOURCEPOSITIONDATA_H
#define DATAFORMATS_HCALRECHIT_HCALSOURCEPOSITIONDATA_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

class HcalSourcePositionData {
public:

  HcalSourcePositionData();
  ~HcalSourcePositionData(){}
  
  void setGlobal(int glStatus, int maxDrivers);

  void setDriver(int message_counter,
		 int timestamp1_sec,
		 int timestamp1_usec,
		 int timestamp2_sec,
		 int timestamp2_usec,
		 int status,
		 int index_counter,
		 int reel_counter,
		 int motor_current,
		 int speed,
		 int tube_number,
		 int motor_number);

  inline int getGlobalStatus() const { return globalStatus; }
  inline int getMaxDrivers() const { return maxDrivers; }
 
  inline int getMessageCounter() const { return message_counter; }
  inline int getStatus() const { return status; }
  inline int getIndexCounter() const { return index_counter; }
  inline int getReelCounter() const { return reel_counter; } 
  inline int getMotorCurrent() const { return motor_current; }
  inline int getSpeed() const { return speed; } 
  inline int getTubeNumber() const { return tube_number; }
  inline int getMotorNumber() const { return motor_number; }
  void getDriverTimestamp(int& seconds, int& useconds) const;
  void getDAQTimestamp(int& seconds, int& useconds) const;

private:

  int globalStatus;
  int maxDrivers;

  int message_counter;
  int index_counter;
  int reel_counter;
  int timestamp1_sec;
  int timestamp1_usec;
  int timestamp2_sec;
  int timestamp2_usec;
  int status;
  int motor_current;
  int speed;
  int tube_number;
  int motor_number;

};

std::ostream& operator<<(std::ostream& s, const HcalSourcePositionData& hspd);

#endif
