#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
//
#include <initializer_list>
#include <vector>
#include <map>

namespace cond {

  namespace time {
    static const std::pair<const char*, TimeType> s_timeTypeMap[] = { std::make_pair("Run", cond::runnumber),
                                                                      std::make_pair("Time", cond::timestamp ),
                                                                      std::make_pair("Lumi", cond::lumiid ),
                                                                      std::make_pair("Hash", cond::hash ),
                                                                      std::make_pair("User", cond::userid ) };
    std::string timeTypeName(TimeType type) {
      if( type==invalid ) return "";
      return s_timeTypeMap[type].first;
    }
    
    TimeType timeTypeFromName( const std::string& name ){
      for (auto const &i : s_timeTypeMap)
        if (name.compare(i.first) == 0)
          return i.second;
      const cond::TimeTypeSpecs & theSpec = cond::findSpecs( name );
      return theSpec.type;
      //throwException( "TimeType \""+name+"\" is unknown.","timeTypeFromName");
    }

    Time_t tillTimeFromNextSince( Time_t nextSince, TimeType timeType ){
      if( timeType != (TimeType)TIMESTAMP ){
	return nextSince - 1;
      } else {
	UnpackedTime unpackedTime = unpack(  nextSince );
	//number of seconds in nanoseconds (avoid multiply and divide by 1e09)
	Time_t totalSecondsInNanoseconds = ((Time_t)unpackedTime.first)*1000000000;
	//total number of nanoseconds
	Time_t totalNanoseconds = totalSecondsInNanoseconds + ((Time_t)(unpackedTime.second));
	//now decrementing of 1 nanosecond
	totalNanoseconds--;
	//now repacking (just change the value of the previous pair)
	unpackedTime.first = (unsigned int) (totalNanoseconds/1000000000);
	unpackedTime.second = (unsigned int)(totalNanoseconds - (Time_t)unpackedTime.first*1000000000);
	return pack(unpackedTime);
      }
    }

    // framework conversions
    edm::IOVSyncValue toIOVSyncValue( Time_t time, TimeType timetype, bool startOrStop) {
      switch (timetype) {
      case RUNNUMBER :
	return edm::IOVSyncValue( edm::EventID(time,
					       startOrStop ? 0 : edm::EventID::maxEventNumber(),
					       startOrStop ? 0 : edm::EventID::maxEventNumber())
				  );
      case LUMIID :
	{
	  edm::LuminosityBlockID l(time);
	  return edm::IOVSyncValue(edm::EventID(l.run(),
						l.luminosityBlock(),
						startOrStop ? 0 : edm::EventID::maxEventNumber())
				   );
	}
      case TIMESTAMP :
	return edm::IOVSyncValue( edm::Timestamp(time));
      default:
	return  edm::IOVSyncValue::invalidIOVSyncValue();
      }
    }

    Time_t fromIOVSyncValue(edm::IOVSyncValue const & time, TimeType timetype) {
      switch (timetype) {
      case RUNNUMBER :
	return time.eventID().run();
      case LUMIID :
	{
	  edm::LuminosityBlockID lum(time.eventID().run(), time.luminosityBlockNumber());
	  return lum.value();
	}
      case TIMESTAMP :
	return time.time().value();
      default:
	return 0;
      }
    }

    // the minimal maximum-time an IOV can extend to                                                                                                                                                
    edm::IOVSyncValue limitedIOVSyncValue( Time_t time, TimeType timetype) {
      switch (timetype) {
      case RUNNUMBER :
	// last lumi and event of this run
	return edm::IOVSyncValue( edm::EventID(time,
					       edm::EventID::maxEventNumber(),
					       edm::EventID::maxEventNumber())
				  );
      case LUMIID :
	{
	  // the same lumiblock
	  edm::LuminosityBlockID l(time);
	  return edm::IOVSyncValue(edm::EventID(l.run(),
						l.luminosityBlock(),
						edm::EventID::maxEventNumber())
				   );
	}
      case TIMESTAMP :
	// next event ?
	return edm::IOVSyncValue::invalidIOVSyncValue();
      default:
	return  edm::IOVSyncValue::invalidIOVSyncValue();
      }
    }

    edm::IOVSyncValue limitedIOVSyncValue(edm::IOVSyncValue const & time, TimeType timetype) {
      switch (timetype) {
      case RUNNUMBER :
	// last event of this run
	return edm::IOVSyncValue(edm::EventID(time.eventID().run(),
					      edm::EventID::maxEventNumber(),
					      edm::EventID::maxEventNumber())
				 );
      case LUMIID :
	// the same lumiblock
	return edm::IOVSyncValue(edm::EventID(time.eventID().run(),
					      time.luminosityBlockNumber(),
					      edm::EventID::maxEventNumber())
				 );
      case TIMESTAMP :
	// same lumiblock
	return edm::IOVSyncValue(edm::EventID(time.eventID().run(),
					      time.luminosityBlockNumber(),
					      edm::EventID::maxEventNumber())
				 );
      default:
	return  edm::IOVSyncValue::invalidIOVSyncValue();
      }
    }

  }

}
