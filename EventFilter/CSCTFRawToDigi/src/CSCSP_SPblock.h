#ifndef CSCSP_SPblock_h
#define CSCSP_SPblock_h
#include <vector>
#include <cstring>
#include "EventFilter/CSCTFRawToDigi/src/CSCSP_MEblock.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCSP_MBblock.h"

class CSCSP_SPblock {
private:
	/////// word 1 ///////
	unsigned phi_        : 5; // azimuth coordinate
	unsigned sign_       : 1; // deltaPhi sign bit, part of the PT LUT address
	unsigned front_rear  : 1; // front/rear bit
	unsigned charge_     : 1; // muon charge bit
	unsigned eta_        : 5; // pseudo rapidity, eta[4:1] is a part of the PT LUT address
	unsigned halo_       : 1; // halo bit
	unsigned se          : 1; // synchronization error: OR of 15 SM bits for all LCTs and similar bits for 2 MB Stubs, passed to the MS
	unsigned zero_1      : 1; // format specific
	/////// word 2 ///////
	unsigned deltaPhi12_ : 8; // difference in phi between station 1 and 2, part of the PT LUT address
	unsigned deltaPhi23_ : 4; // difference in phi between station 2 and 3, part of the PT LUT address
	unsigned zero_2      : 1; // format specific
	unsigned bxn0_       : 1; // OR of BX0 signals received with ME LCTs and MB stubs, passed to the MS
	unsigned bc0_        : 1; // OR of BC0 signals received with ME LCTs and MB stubs, passed to the MS
	unsigned zero_3      : 1; // format specific
	/////// word 3 ///////
	unsigned me1_id      : 3; // track stubs used to build up the track
	unsigned me2_id      : 2; // ...
	unsigned me3_id      : 2; // ...
	unsigned me4_id      : 2; // ...
	unsigned mb_id       : 3; // ...
	unsigned ms_id       : 3; // Muon Sorter Winner bit positional code
	unsigned zero_4      : 1; // format specific
	/////// word 4 ///////
	unsigned me1_tbin    : 3; // time bins of the above track stubs used to build up a track
	unsigned me2_tbin    : 3; // ...
	unsigned me3_tbin    : 3; // ...
	unsigned me4_tbin    : 3; // ...
	unsigned mb_tbin     : 3; //
	unsigned zero_5      : 1; // format specific

	// LCTs and MB stub, that formed this track should be easily accessible through the track interface
	//  Hence we keep copies of the data here and let top-level unpacking set these data
	friend class CSCSPEvent;
	CSCSP_MEblock lct_[4]; // LCTs from four stations
	CSCSP_MBblock dt_;     // MB stub
	// LCTs and stub were used (=true) in this record
	bool lctFilled[4], dtFilled;

	// Other data members logically belong to SP Block record,
	//  but physically are located in Data Block Header, which implementation is:
	friend class CSCSPRecord;
	friend class CSCTFPacker;
	unsigned int tbin_; // time bin, that this SP block belongs to
	unsigned int mode_; // stations, that this track crossed (they gave LCTs to build it)
	unsigned int id_;   // track number (1, 2, or 3)

public:
	bool check(void) const throw() { return zero_1!=0||zero_2!=0||zero_3!=0||zero_4!=0||zero_5!=0; }

	unsigned int phi   (void) const throw() { return phi_;    }
	unsigned int sign  (void) const throw() { return sign_;   }
	unsigned int f_r   (void) const throw() { return front_rear; };
	unsigned int charge(void) const throw() { return charge_; }
	unsigned int eta   (void) const throw() { return eta_;    }
	unsigned int halo  (void) const throw() { return halo_;   }
	unsigned int syncErr(void)const throw() { return se;      }

	unsigned int deltaPhi12(void) const throw() { return deltaPhi12_; }
	unsigned int deltaPhi23(void) const throw() { return deltaPhi23_; }
	unsigned int bx0       (void) const throw() { return bxn0_;   }
	unsigned int bc0       (void) const throw() { return bc0_;    }

	unsigned int ME1_id(void) const throw() { return me1_id; }
	unsigned int ME2_id(void) const throw() { return me2_id; }
	unsigned int ME3_id(void) const throw() { return me3_id; }
	unsigned int ME4_id(void) const throw() { return me4_id; }
	unsigned int MB_id (void) const throw() { return mb_id;  }
	unsigned int MS_id (void) const throw() { return ms_id;  }

	unsigned int ME1_tbin(void) const throw() { return me1_tbin; }
	unsigned int ME2_tbin(void) const throw() { return me2_tbin; }
	unsigned int ME3_tbin(void) const throw() { return me3_tbin; }
	unsigned int ME4_tbin(void) const throw() { return me4_tbin; }
	unsigned int MB_tbin (void) const throw() { return mb_tbin;  }

	unsigned int tbin(void) const throw() { return tbin_; }
	unsigned int id  (void) const throw() { return id_;   }

	// vector may have up to 4 elements (one per station)
	std::vector<CSCSP_MEblock> LCTs(void) const throw() {
		std::vector<CSCSP_MEblock> result;
		for(int station=0; station<4; station++)
			if(lctFilled[station]) result.push_back(lct_[station]);
		return result;
	}

	// vector either empty or has one element
	std::vector<CSCSP_MBblock> dtStub(void) const throw() {
		std::vector<CSCSP_MBblock> result;
		if(dtFilled) result.push_back(dt_);
		return result;
	}

	unsigned int ptLUTaddress(void) const throw() { return (sign_<<20) | (mode_<<16) | ((eta_&0x1E)<<11) | (deltaPhi23_<<8) | deltaPhi12_; }
	unsigned int mode        (void) const throw() { return mode_; }

	bool unpack(const unsigned short *&buf) throw() { std::memcpy((void*)this,buf,4*sizeof(short)); buf+=4; return check(); }

	CSCSP_SPblock(void){}
};

#endif
