#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include <cstdio>

using namespace std;

HcalSourcePositionData::HcalSourcePositionData() {
  messageCounter_ = 0;
  indexCounter_ = 0;
  reelCounter_ = 0;
  timestamp1_sec_ = 0;
  timestamp1_usec_ = 0;
  timestamp2_sec_ = 0;
  timestamp2_usec_ = 0;
  status_ = 0;
  motorCurrent_ = 0;
  motorVoltage_ = 0;
  tubeId_ = -1;
  driverId_ = -1;
  sourceId_ = -1;
  tubeNameFromCoord_ = "";
  tubeDescriptionFromSD_ = "";
  lastCommand_ = "";
  message_ = "";
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
                                 int motor_voltage,
                                 int driver_id,
                                 int source_id,
                                 std::string tubeNameFromCoord,
                                 std::string tubeDescFromSD,
                                 std::string lastCommand,
                                 std::string message) {
  messageCounter_ = message_counter;
  indexCounter_ = index_counter;
  reelCounter_ = reel_counter;
  timestamp1_sec_ = timestamp1_sec;
  timestamp1_usec_ = timestamp1_usec;
  timestamp2_sec_ = timestamp2_sec;
  timestamp2_usec_ = timestamp2_usec;
  status_ = status;
  motorCurrent_ = motor_current;
  motorVoltage_ = motor_voltage;
  driverId_ = driver_id;
  sourceId_ = source_id;
  tubeNameFromCoord_ = tubeNameFromCoord;
  tubeDescriptionFromSD_ = tubeDescFromSD;
  lastCommand_ = lastCommand;
  message_ = message;
}

void HcalSourcePositionData::getDriverTimestamp(int& seconds, int& useconds) const {
  seconds = timestamp1_sec_;
  useconds = timestamp1_usec_;
}

void HcalSourcePositionData::getDAQTimestamp(int& seconds, int& useconds) const {
  seconds = timestamp2_sec_;
  useconds = timestamp2_usec_;
}

ostream& operator<<(ostream& s, const HcalSourcePositionData& hspd) {
  s << "  Message Counter       =" << hspd.messageCounter() << endl;
  s << "  Index Counter         =" << hspd.indexCounter() << endl;
  s << "  Reel Counter          =" << hspd.reelCounter() << endl;
  s << "  Status                =" << hex << hspd.status() << dec << endl;
  s << "  Motor Current         =" << hspd.motorCurrent() << endl;
  s << "  Motor Voltage         =" << hspd.motorVoltage() << endl;
  s << "  Tube Id               =" << hspd.tubeId() << endl;
  s << "  Driver Id             =" << hspd.driverId() << endl;
  s << "  Source Id             =" << hspd.sourceId() << endl;
  s << "  TubeNameFromCoord     =" << hspd.tubeNameFromCoord() << endl;
  s << "  TubeDescriptionFromSD =" << hspd.tubeDescriptionFromSD() << endl;
  s << "  Last Command          =" << hspd.lastCommand() << endl;
  s << "  Message               =" << hspd.message() << endl;

  int timebase = 0;
  int timeusec = 0;
  hspd.getDriverTimestamp(timebase, timeusec);
  // trim seconds off of usec and add to base
  timeusec %= 1000000;
  timebase += timeusec / 1000000;
  char str[50];
  sprintf(str, "  Driver Timestamp : %s", ctime((time_t*)&timebase));
  s << str;
  timebase = 0;
  timeusec = 0;
  hspd.getDAQTimestamp(timebase, timeusec);
  timeusec %= 1000000;
  timebase += timeusec / 1000000;

  sprintf(str, "  DAQ Timestamp : %s", ctime((time_t*)&timebase));
  s << str;

  return s;
}
