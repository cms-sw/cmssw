#ifndef CSCSPEvent_h
#define CSCSPEvent_h

#include "EventFilter/CSCTFRawToDigi/src/CSCSPHeader.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPCounters.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPRecord.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPTrailer.h"

class CSCSPEvent {
private:
	CSCSPHeader   header_;
	CSCSPCounters counters_;
	CSCSPRecord   record_[7];
	CSCSPTrailer  trailer_;

public:
	const CSCSPHeader&   header  (void) const throw() { return header_;   }
	const CSCSPCounters& counters(void) const throw() { return counters_; }
	const CSCSPTrailer&  trailer (void) const throw() { return trailer_;  }

	const CSCSPRecord& record(unsigned int tbin) const throw() { return record_[tbin]; }

	bool unpack(const unsigned short *&buf) throw() ;

	CSCSPEvent(void){}
};

#endif
