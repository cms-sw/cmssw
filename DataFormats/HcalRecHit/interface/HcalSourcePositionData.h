#ifndef DATAFORMATS_HCALRECHIT_HCALSOURCEPOSITIONDATA_H
#define DATAFORMATS_HCALRECHIT_HCALSOURCEPOSITIONDATA_H 1

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

class HcalSourcePositionData {
public:

  HcalSourcePositionData();
  ~HcalSourcePositionData(){}
  
  inline int messageCounter() const { return messageCounter_; }
  inline int status() const { return status_; }
  inline int indexCounter() const { return indexCounter_; }
  inline int reelCounter() const { return reelCounter_; } 
  inline int motorCurrent() const { return motorCurrent_; }
  inline int speed() const { return speed_; } 
  inline int tubeId() const { return tubeId_; }
  inline int driverId() const { return driverId_; }
  inline int sourceId() const { return sourceId_; }

  void getDriverTimestamp(int& seconds, int& useconds) const;
  void getDAQTimestamp(int& seconds, int& useconds) const;

  void set(      int message_counter,
		 int timestamp1_sec,
		 int timestamp1_usec,
		 int timestamp2_sec,
		 int timestamp2_usec,
		 int status,
		 int index_counter,
		 int reel_counter,
		 int motor_current,
		 int speed,
		 int tube_id,
		 int driver_id,
                 int source_id);
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
  int speed_;
  int tubeId_;
  int driverId_;
  int sourceId_;
};

std::ostream& operator<<(std::ostream& s, const HcalSourcePositionData& hspd);

#endif
