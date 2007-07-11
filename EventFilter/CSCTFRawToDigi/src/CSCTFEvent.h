#ifndef CSCTFEvent_h
#define CSCTFEvent_h

#include "EventFilter/CSCTFRawToDigi/src/CSCSPEvent.h"
#include <vector>

class CSCTFEvent {
private:
	CSCSPEvent sp[12];
	int nRecords;

public:
	// Before we do unpacking, we need to do basic TF format checks (TF Binary Examiner)
	enum {MISSING_HEADER=0x2, MISSING_TRAILER=0x4, OUT_OF_BUFFER=0x8, WORD_COUNT=0x10, CONFIGURATION=0x20};

	std::vector<CSCSPEvent> SPs(void) const throw() {
		std::vector<CSCSPEvent> result;
		for(int spNum=0; spNum<nRecords; spNum++) result.push_back(sp[spNum]);
		return result;
	}

	unsigned int unpack(const unsigned short *buf, unsigned int length) throw() ;

	CSCTFEvent(void){}
};

#endif
