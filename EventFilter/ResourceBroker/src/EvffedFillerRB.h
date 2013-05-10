#ifndef EVENTFILTER_RESOURCEBROKER_EVFFEDFILLERRB
#define EVENTFILTER_RESOURCEBROKER_EVFFEDFILLERRB

#include <unistd.h>

#include "interface/shared/fed_header.h"

#define FED_HCTRLID_INSERT          ( ( (FED_SLINK_START_MARKER) & FED_HCTRLID_WIDTH )<< FED_HCTRLID_SHIFT) 
#define FED_EVTY_INSERT(a)          ( ( (a) & FED_EVTY_WIDTH )	       << FED_EVTY_SHIFT         ) 
#define FED_LVL1_INSERT(a)          ( ( (a) & FED_LVL1_WIDTH )	       << FED_LVL1_SHIFT         ) 
#define FED_BXID_INSERT(a)          ( ( (a) & FED_BXID_WIDTH )	       << FED_BXID_SHIFT         ) 
#define FED_SOID_INSERT(a)          ( ( (a) & FED_SOID_WIDTH )	       << FED_SOID_SHIFT         ) 
#define FED_VERSION_INSERT(a)       ( ( (a) & FED_VERSION_WIDTH )      << FED_VERSION_SHIFT      ) 
#define FED_MORE_HEADERS_INSERT(a)  ( ( (a) & FED_MORE_HEADERS_WIDTH ) << FED_MORE_HEADERS_SHIFT ) 

#include "interface/shared/fed_trailer.h"
#define FED_TCTRLID_INSERT          ( ( (FED_SLINK_END_MARKER) & FED_TCTRLID_WIDTH ) << FED_TCTRLID_SHIFT )       
#define FED_EVSZ_INSERT(a)          ( ( (a) & FED_EVSZ_WIDTH )	        << FED_EVSZ_SHIFT )          
#define FED_CRCS_INSERT(a)          ( ( (a) & FED_CRCS_WIDTH )	        << FED_CRCS_SHIFT )          
#define FED_STAT_INSERT(a)          ( ( (a) & FED_STAT_WIDTH )	        << FED_STAT_SHIFT )          
#define FED_TTSI_INSERT(a)          ( ( (a) & FED_TTSI_WIDTH )	        << FED_TTSI_SHIFT )          
#define FED_MORE_TRAILERS_INSERT(a) ( ( (a) & FED_MORE_TRAILERS_WIDTH ) << FED_MORE_TRAILERS_SHIFT ) 

#include "EventFilter/FEDInterface/interface/FED1023.h"
#include "EventFilter/ResourceBroker/interface/FUResourceBroker.h"

namespace evf {

class EvffedFillerRB {

public:
	EvffedFillerRB(FUResourceBroker *rb) {
		for (unsigned int i = 0; i < fedinterface::EVFFED_LENGTH; i++) {
			*(payload_.asWords + i) = 0;
		}
		char hostname[32];
		int retval = gethostname(hostname, 32);
		if (retval != 0) {
			hostid_ = 0xdead;
		} else {
			if (strtok(hostname, "-") == 0)
				hostid_ = 0xdead;
			char *p = strtok(0, "-"); // rack id
			long hostid = 0xdead;
			if (p != 0)
				hostid = strtol(p, 0, 16) << 8;
			p = strtok(0, "-"); // node id
			if (p == 0)
				hostid += 0xdead;
			else
				hostid += strtol(p, 0, 16);
			hostid_ = hostid;
		}
		*(payload_.asHWords + (fedinterface::EVFFED_RBIDENT_OFFSET/sizeof(uint32_t)))
				= ((hostid_ & fedinterface::EVFFED_RBPCIDE_MASK)
						<< fedinterface::EVFFED_RBPCIDE_SHIFT)
						+ ((rb->instanceNumber()
								& fedinterface::EVFFED_RBINSTA_MASK)
								<< fedinterface::EVFFED_RBINSTA_SHIFT);
	}
	unsigned char * const getPayload() {
		return payload_.asBytes;
	}
	uint32_t getSize() {
		return fedinterface::EVFFED_TOTALSIZE;
	}

	void putHeader(unsigned int l1id, unsigned int bxid) {
		*(payload_.asHWords) = FED_SOID_INSERT(fedinterface::EVFFED_ID)
				+ FED_VERSION_INSERT(fedinterface::EVFFED_VERSION);
		*(payload_.asHWords + (evtn::SLINK_HALFWORD_SIZE/sizeof(uint32_t)))
				= FED_HCTRLID_INSERT + FED_EVTY_INSERT(0x1)
						+ FED_LVL1_INSERT(l1id) + FED_BXID_INSERT(bxid);

	}
	// this function MUST be called again after filling is complete (hence again in EP!!!)
	void putTrailer() {
		uint32_t *fedtr_p = payload_.asHWords
				+ ((fedinterface::EVFFED_TOTALSIZE - evtn::FED_TRAILER_SIZE)/sizeof(uint32_t));
		*(fedtr_p + (evtn::SLINK_HALFWORD_SIZE/sizeof(uint32_t))) = FED_TCTRLID_INSERT
				+ FED_EVSZ_INSERT(fedinterface::EVFFED_LENGTH);
		*(fedtr_p) = FED_CRCS_INSERT(compute_crc(payload_.asBytes,fedinterface::EVFFED_TOTALSIZE));
	}
	void setRBTimeStamp(uint64_t ts) {
		*(payload_.asWords + (fedinterface::EVFFED_RBWCTIM_OFFSET/sizeof(uint64_t)))
				= ts;
	}
	void setRBEventCount(uint32_t evtcnt) {
		*(payload_.asHWords + (fedinterface::EVFFED_RBEVCNT_OFFSET/sizeof(uint32_t)))
				= evtcnt;
	}

	void setEPProcessId(pid_t pid) {
		*(payload_.asHWords + (fedinterface::EVFFED_EPIDENT_OFFSET/sizeof(uint32_t)))
				= (pid & fedinterface::EVFFED_EPPCIDE_MASK)
						<< fedinterface::EVFFED_EPPCIDE_SHIFT;
	}
	unsigned int fedId() const {
		return fedinterface::EVFFED_ID;
	}
	unsigned int size() const {
		return fedinterface::EVFFED_TOTALSIZE;
	}
private:
	union Payload {
		unsigned char asBytes[fedinterface::EVFFED_TOTALSIZE];
		uint32_t asHWords[fedinterface::EVFFED_TOTALSIZE / sizeof(uint32_t)];
		uint64_t asWords[fedinterface::EVFFED_TOTALSIZE / sizeof(uint64_t)];
	};
	Payload payload_;
	unsigned int hostid_;
};
}

#endif
