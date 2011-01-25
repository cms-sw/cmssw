#ifndef CSCSPRecord_h
#define CSCSPRecord_h
#include <vector>

#include "EventFilter/CSCTFRawToDigi/src/CSCSP_MEblock.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCSP_MBblock.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCSP_SPblock.h"

class CSCSPRecord {
private:
	// Data Block Header format
	/////// word 1 ///////
	unsigned vp_bits  : 15; // Valid Pattern bits for 15 ME LCTs
	unsigned zero_1   : 1;  // format specific
	/////// word 2 ///////
	unsigned mode1    : 4;  // First track mode (ids of stations, that give LCTs to build a track, mode=0 - no track)
	unsigned mode2    : 4;  // Second track mode (...)
	unsigned mode3    : 4;  // Third track mode (...)
	unsigned vq_a     : 1;  // Valid Quality (Q > 0) bit for the first MB Stub;
	unsigned vq_b     : 1;  // Valid Quality (Q > 0) bit for the second MB Stub;
	unsigned spare_1  : 1;  // not used
	unsigned zero_2   : 1;  // format specific
	/////// word 3 ///////
	unsigned se_bits  : 15; // Sync Error bits for 15 ME LCTs
	unsigned zero_3   : 1;  // format specific
	/////// word 4 ///////
	unsigned sm_bits  : 15; // Modified Sync Error bits for 15 ME LCTs: (SE) OR (Optical Link status) OR (Alignment FIFO status) OR (Bunch Crossing Counter status)
	unsigned zero_4   : 1;  // format specific
	/////// word 5 ///////
	unsigned af_bits  : 15; // Alignment FIFO status bits for 15 ME links, in normal state zero
	unsigned zero_5   : 1;  // format specific
	/////// word 6 ///////
	unsigned bx_bits  : 15; // set for LCT with BC0 commig early or later that the BC0 => monitors the ALCT/TMB/MPC timing
	unsigned zero_6   : 1;  // format specific
	/////// word 7 ///////
	unsigned pt_low      : 8; // low byte of pt for track, that we are spying at
	unsigned pt_spy_point: 2; // defines track, that we are spying at
	unsigned spare_2     : 1; // not used
	unsigned spare_3     : 1; // not used
	unsigned af_barrel_1 : 1;
	unsigned af_barrel_2 : 1;
	unsigned spare_4     : 1; // not used
	unsigned zero_7      : 1; // format specific
	/////// word 8 ///////
	unsigned pt_high     : 8; // high byte of pt for track, that we are spying at
        unsigned time_bin    : 3;
  
//	unsigned spare_5     : 1; // not used
//	unsigned spare_6     : 1; // not used
//	unsigned spare_7     : 1; // not used

	unsigned spare_8     : 1; // not used
	unsigned bx_barrel_1 : 1;
	unsigned bx_barrel_2 : 1;
	unsigned spare_9     : 1; // not used
	unsigned zero_8      : 1; // format specific

	// persistent storage for specific data blocks
	CSCSP_MEblock me[5][3];
	CSCSP_MBblock mb[2];
	CSCSP_SPblock sp[3];
	// data blocks, that were used (=true) in this record
	bool meFilled[5][3], mbFilled[2], spFilled[3];
	// Allow CSCSPEvent, front end to the data format, access 'sp' field dirrectly
	friend class CSCSPEvent;
	friend class CSCTFPacker;

public:
	bool check(void) const throw() {
		return zero_1 !=0 || zero_2 !=0 || zero_3 !=0 || zero_4 !=0
			|| zero_5 !=0 || zero_6 !=0 || zero_7 !=0 || zero_8 !=0 // || spare_1!=0
			|| spare_2!=0 || spare_3!=0 || spare_4!=0 || spare_8!=0
		        || spare_9!=0;
	}

	// Following functions return empty vector if no LCTs/MB_stubs/tracks are available
	std::vector<CSCSP_MEblock> LCTs   (void) const throw();
	std::vector<CSCSP_MEblock> LCTs   (unsigned int mpc) const throw();
	std::vector<CSCSP_MEblock> LCT    (unsigned int mpc, unsigned int link) const throw();
	std::vector<CSCSP_SPblock> tracks (void) const throw();
	std::vector<CSCSP_MBblock> mbStubs(void) const throw();

	unsigned int VPs(void) const throw() { return vp_bits|(vq_a<<15)|(vq_b<<16); }
	unsigned int SEs(void) const throw() { return se_bits; }
	unsigned int SMs(void) const throw() { return sm_bits; }
	unsigned int AFs(void) const throw() { return af_bits|(af_barrel_1<<15)|(af_barrel_2<<16); }
	unsigned int BXs(void) const throw() { return bx_bits|(bx_barrel_1<<15)|(bx_barrel_2<<16); }

	unsigned int ptSpy     (void) const throw() { return (pt_high<<8)|pt_low; }
	unsigned int ptSpyTrack(void) const throw() { return pt_spy_point;        }

	bool unpack(const unsigned short* &buf, unsigned int nonmasked_data_blocks, bool empty_blocks_suppressed, unsigned int tbin) throw() ;

	CSCSPRecord(void){}
};

#endif
