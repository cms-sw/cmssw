#include "CondCore/IOVService/interface/migrateIOV.h"
#include "IOV.h"
#include "CondFormats/Common/interface/IOVSequence.h"
#include <algorithm>
#include <boost/bind.hpp>


namespace cond {


  IOVSequence * migrateIOV(IOV const & iov) {
    IOVSequence * result = new IOVSequence(iov.timetype,iov.firstsince,"");
    (*result).iovs().reserve(iov.iov.size());
    std::for_each(iov.iov.begin(),iov.iov.end(),
		  boost::bind(&IOVSequence::add,result,
			      boost::bind(&IOV::Item::first,_1),
			      boost::bind(&IOV::Item::second,_1),
			      std::string("")
			      )
		  );
      return result;
  }





  IOV * backportIOV(IOVSequence const & sequence) {
    IOV * result = new IOV(sequence.timeType(), sequence.firstsince());
    (*result).iov.reserve(sequence.iovs().size());
    std::for_each(sequence.iovs().begin(),sequence.iovs().end(),
		  boost::bind(&IOV::add,result,
			      boost::bind(&IOVSequence::Item::tillTime,_1),
			      boost::bind(&IOVSequence::Item:: payloadToken,_1)
			      )
		  );
      return result;
  }




}
