#include "CondCore/DBCommon/interface/ConvertIOVSyncValue.h"



namespace cond {

  edm::IOVSyncValue toIOVSyncValue(cond::Time_t time, cond::TimeType timetype, bool startOrStop) {
    switch (timetype) {
    case cond::runnumber :
      return edm::IOVSyncValue( edm::EventID(time, 
					     startOrStop ? 0 : edm::EventID::maxEventNumber()) );
    case cond::lumiid :
	return edm::IOVSyncValue(edm::EventID(l.run(),
					      startOrStop ? 0 : edm::EventID::maxEventNumber()), 
				 edm::LuminosityBlockID(time).luminosityBlock());
    case cond::timestamp :
      return edm::IOVSyncValue( edm::Timestamp(time));
    default:
      return  edm::IOVSyncValue::invalidIOVSyncValue();
    }
  }

  cond::Time_t fromIOVSyncValue(edm::IOVSyncValue const & time, cond::TimeType timetype) {
    switch (timetype) {
    case cond::runnumber :
      return time.eventID().run();
    case cond::lumiid : 
      {
	edm::LuminosityBlockID lum(time.eventID().run(), time.luminosityBlockNumber());
	return lum.value();
      }
    case cond::timestamp :
      return time.time().value();
    default:
      return 0;
    }
  }

  // the minimal maximum-time an IOV can extend to
  edm::IOVSyncValue limitedIOVSyncValue(cond::Time_t time, cond::TimeType timetype) {
    switch (timetype) {
    case cond::runnumber :
      // last event of this run
      return edm::IOVSyncValue( edm::EventID(time,edm::EventID::maxEventNumber()) );
    case cond::lumiid : 
      {
	// the same lumiblock
	edm::LuminosityBlockID l(time);
	return edm::IOVSyncValue(edm::EventID(l.run(), edm::EventID::maxEventNumber()), 
				 l.luminosityBlock());
      }
    case cond::timestamp :
      // next event ?
      return edm::IOVSyncValue::invalidIOVSyncValue();
    default:
      return  edm::IOVSyncValue::invalidIOVSyncValue();
    }
  }

    edm::IOVSyncValue limitedIOVSyncValue(edm::IOVSyncValue const & time, cond::TimeType timetype) {
      switch (timetype) {
      case cond::runnumber :
	// last event of this run
	return edm::IOVSyncValue(edm::EventID(time.eventID().run(),edm::EventID::maxEventNumber()) );
      case cond::lumiid :
	// the same lumiblock
	return edm::IOVSyncValue(edm::EventID(time.eventID().run(),edm::EventID::maxEventNumber()),
				 time.luminosityBlockNumber());
      case cond::timestamp :
	// same lumiblock
	return edm::IOVSyncValue(edm::EventID(time.eventID().run(),edm::EventID::maxEventNumber()),
				 time.luminosityBlockNumber());	
      default:
	return  edm::IOVSyncValue::invalidIOVSyncValue();
      }
    }



  }

}

