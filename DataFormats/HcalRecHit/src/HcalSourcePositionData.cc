#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include <cstdio>

using namespace std;

HcalSourcePositionData::HcalSourcePositionData(){
  messageCounter_=0;
  indexCounter_=0;
  reelCounter_=0;
  timestamp1_sec_=0;
  timestamp1_usec_=0;
  timestamp2_sec_=0;
  timestamp2_usec_=0;
  status_=0;
  motorCurrent_=0;
  speed_=0;
  tubeId_=-1;
  driverId_=-1;
  sourceId_=-1;
}

void HcalSourcePositionData::set(int message_counter,
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
                                       int source_id){

  messageCounter_=message_counter;
  indexCounter_=index_counter;
  reelCounter_=reel_counter;
  timestamp1_sec_=timestamp1_sec;
  timestamp1_usec_=timestamp1_usec;
  timestamp2_sec_=timestamp2_sec;
  timestamp2_usec_=timestamp2_usec;
  status_=status;
  motorCurrent_=motor_current;
  speed_=speed;
  tubeId_=tube_id;
  driverId_=driver_id;
  sourceId_=source_id;
}

void HcalSourcePositionData::getDriverTimestamp(int& seconds, int& useconds) const{
  seconds=timestamp1_sec_;
  useconds=timestamp1_usec_;
}

void HcalSourcePositionData::getDAQTimestamp(int& seconds, int& useconds) const{
  seconds=timestamp2_sec_;
  useconds=timestamp2_usec_;
}

ostream& operator<<(ostream& s, const HcalSourcePositionData& hspd) {

  s << "  Message Counter =" << hspd.messageCounter() << endl;
  s << "  Index Counter   =" << hspd.indexCounter() << endl;
  s << "  Reel Counter    =" << hspd.reelCounter() << endl;
  s << "  Status          =" << hex << hspd.status() << dec << endl;
  s << "  Motor Current   =" << hspd.motorCurrent() << endl;
  s << "  Speed           =" << hspd.speed() << endl;
  s << "  Tube Id         =" << hspd.tubeId() << endl;
  s << "  Driver Id       =" << hspd.driverId() << endl;
  s << "  Source Id       =" << hspd.sourceId() << endl;
  
  int timebase =0; int timeusec=0;
  hspd.getDriverTimestamp(timebase,timeusec);
  // trim seconds off of usec and add to base
  timeusec %= 1000000;
  timebase += timeusec/1000000;
  char str[50];
  sprintf(str, "  Driver Timestamp : %s", ctime((time_t *)&timebase));
  s << str;
  timebase=0; timeusec=0;
  hspd.getDAQTimestamp(timebase,timeusec);
  timeusec %= 1000000;
  timebase += timeusec/1000000;
  
  sprintf(str, "  DAQ Timestamp : %s", ctime((time_t *)&timebase));
  s << str;
  
  return s;
}
