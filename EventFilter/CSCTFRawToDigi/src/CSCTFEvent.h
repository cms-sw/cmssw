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
	enum {MISSING_HEADER=0x2, MISSING_TRAILER=0x4, OUT_OF_BUFFER=0x8, WORD_COUNT=0x10, CONFIGURATION=0x20, NONSENSE=0x40};

	std::vector<CSCSPEvent> SPs(void) const throw() {
		std::vector<CSCSPEvent> result;
                result.reserve(nRecords);
		for(int spNum=0; spNum<nRecords; spNum++) result.push_back(sp[spNum]);
		return result;
	}

	// Faster analog of the previous function:
    std::vector<const CSCSPEvent*> SPs_fast(void) const throw() {
        std::vector<const CSCSPEvent*> retval;
        retval.clear();
        retval.reserve(nRecords);
        for(int spNum=0; spNum<nRecords; spNum++) retval.push_back(sp+spNum);
        return retval;
    }

	unsigned int unpack(const unsigned short *buf, unsigned int length) throw() ;

	CSCTFEvent(void){}
};

#endif
