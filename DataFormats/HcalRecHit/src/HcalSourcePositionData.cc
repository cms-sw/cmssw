#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

using namespace std;

HcalSourcePositionData::HcalSourcePositionData(){
  globalStatus=0;
  maxDrivers=0;
    message_counter=0;
    index_counter=0;
    reel_counter=0;
    timestamp1_sec=0;
    timestamp1_usec=0;
    timestamp2_sec=0;
    timestamp2_usec=0;
    status=0;
    motor_current=0;
    speed=0;
    tube_number=0;
    motor_number=0;
}

void HcalSourcePositionData::setGlobal(int glStatus, int maxD){

  globalStatus = glStatus;
  maxDrivers = maxD;

  return;
}

void HcalSourcePositionData::setDriver(int m_message_counter,
				       int m_timestamp1_sec,
				       int m_timestamp1_usec,
				       int m_timestamp2_sec,
				       int m_timestamp2_usec,
				       int m_status,
				       int m_index_counter,
				       int m_reel_counter,
				       int m_motor_current,
				       int m_speed,
				       int m_tube_number,
				       int m_motor_number){

    message_counter=m_message_counter;
    index_counter=m_index_counter;
    reel_counter=m_reel_counter;
    timestamp1_sec=m_timestamp1_sec;
    timestamp1_usec=m_timestamp1_usec;
    timestamp2_sec=m_timestamp2_sec;
    timestamp2_usec=m_timestamp2_usec;
    status=m_status;
    motor_current=m_motor_current;
    speed=m_speed;
    tube_number=m_tube_number;
    motor_number=m_motor_number;
  
  return;
}

void HcalSourcePositionData::getDriverTimestamp(int& seconds, int& useconds) const{
  seconds=timestamp1_sec;
  useconds=timestamp1_usec;
  return;
}

void HcalSourcePositionData::getDAQTimestamp(int& seconds, int& useconds) const{
  seconds=timestamp2_sec;
  useconds=timestamp2_usec;
  return;
}

ostream& operator<<(ostream& s, const HcalSourcePositionData& hspd) {

  s << "  Global Status: " << hspd.getGlobalStatus() << endl;
  s << "  Drivers: " << hspd.getMaxDrivers() << endl;

  s << "  Message Counter =" << hspd.getMessageCounter() << endl;
  s << "  Index Counter   =" << hspd.getIndexCounter() << endl;
  s << "  Reel Counter    =" << hspd.getReelCounter() << endl;
  s << "  Status          =" << hspd.getStatus() << endl;
  s << "  Motor Current   =" << hspd.getMotorCurrent() << endl;
  s << "  Speed           =" << hspd.getSpeed() << endl;
  s << "  Tube Number     =" << hspd.getTubeNumber() << endl;
  s << "  Motor Number    =" << hspd.getMotorNumber() << endl;
  
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
