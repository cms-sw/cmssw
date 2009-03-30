#include "CondCore/IOVService/interface/migrateIOV.h"
#include "IOV.h"
#include "CondFormats/Common/interface/IOVSequence.h"
#include <algorithm>
#include <boost/bind.hpp>


namespace cond {

  IOVSequence * migrateIOV(IOV const & iov) {
    IOVSequence * result = new IOVSequence(iov.timetype,iov.iov.back().first,"");
    // (*result).iovs().reserve(iov.iov.size());
    cond::Time_t since = iov.firstsince;
    for(IOV::const_iterator p=iov.iov.begin(); p!=iov.iov.end();p++) {
      (*result).add(since, (*p).second);
      since = (*p).first+1;
    }
    return result;
  }
  
  
  IOV * backportIOV(IOVSequence const & sequence) {
    IOV * result = new IOV(sequence.timeType(), sequence.firstSince());
    (*result).iov.reserve(sequence.iovs().size());
    for(IOVSequence::const_iterator p=sequence.iovs().begin();
	  p!=sequence.iovs().end()-1; p++) {
      cond::Time_t  till = (*(p+1)).sinceTime()-1;
      (*result).add(till, (*p).wrapperToken());
    }
    (*result).add(sequence.lastTill(),sequence.iovs().back().wrapperToken());
    return result;
  }

}
