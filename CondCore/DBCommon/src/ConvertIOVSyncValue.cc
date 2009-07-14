#include "CondCore/DBCommon/interface/ConvertIOVSyncValue.h"



namespace cond {

  edm::IOVSyncValue toIOVSyncValue(cond::Time_t time, cond::TimeType timetype, bool startOrStop) {
    switch (timetype) {
    case cond::runnumber :
      return edm::IOVSyncValue( edm::EventID(time,0) );
    case cond::lumiid :
      edm::LuminosityBlockID l(time);
      return edm::IOVSyncValue(edm::EventID(l.run(),startOrStop ? 0 : edm::EventID::maxEventNumber()), 
			       l.luminosityBlock());
    case cond::timestamp :
      return edm::IOVSyncValue( edm::Timestamp(time));
    default:
      return  edm::IOVSyncValue::invalidIOVSyncValue();
    }
  }

  cond::Time_t fromIOVSyncValue(edm::IOVSyncValue const & time) {
    switch (timetype) {
    case cond::runnumber :
      return time.eventID().run();
    case cond::lumiid :
      edm::LuminosityBlockID lum(time.eventID().run(), time.luminosityBlockNumber());
      return lum.value();
    case cond::timestamp :
      return time.time().value();
    default:
      return 0;
    }
  }
  
}

