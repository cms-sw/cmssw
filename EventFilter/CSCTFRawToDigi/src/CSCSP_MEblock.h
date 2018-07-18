#ifndef CSCSP_MEblock_h
#define CSCSP_MEblock_h

#include <cstring>

class CSCSP_MEblock {
private:
	/////// word 1 ///////
	unsigned clct_pattern_number : 4; // higher pattern number - straighter high-momentum tracks with more layers hit, also encodes half/di-strip indication
	unsigned quality_            : 4; // the more hits the higher LCT Quality
	unsigned wire_group_id       : 7; // radial position of the pattern within the chamber (0-111)
	unsigned zero_1              : 1; // format specific
	/////// word 2 ///////
	unsigned clct_pattern_id     : 8; // azimuthal position ot the pattern at the third (key) layer: (0-159 for half-strips 0-39 for di-strips)
	unsigned csc_id              : 4; // chamber # (1-9)
	unsigned left_right          : 1; // L/R - track is heading towards lower/higher strip number
	unsigned bx0_                : 1; // BX counter least significant bit
	unsigned bc0_                : 1; // BC Zero flag marks that next BXN = 0
	unsigned zero_2              : 1; // format specific
	/////// word 3 ///////
	unsigned me_bxn              : 12;// LCT arrival time picked from a local 12-bit BX Counter, that runs at link timing
	unsigned receiver_status_er1 : 1; // receiver status for the frame 1 (see below)
	unsigned receiver_status_dv1 : 1; // receiver status for the frame 1 (see below)
	unsigned aligment_fifo_full  : 1; // Alignment FIFO Full Flag, should be 0, if AF has been initialized successfully by L1Reset
	unsigned zero_3              : 1; // format specific
	/////// word 4 ///////
	unsigned link_id             : 2; // Link number (1-3)         [reported by MPC on every L1 Reset]
	unsigned mpc_id              : 6; // MPC Crate number (0-63)   [reported by MPC on every L1 Reset]
	unsigned err_prop_cnt        : 4; // accumulates the "Receive Error Propagation" occurrences since last L1Reset
	unsigned receiver_status_er2 : 1; // receiver status for the frame 2 (see below)
	unsigned receiver_status_dv2 : 1; // receiver status for the frame 2 (see below)
	unsigned aligment_fifo_empty : 1; // Alignment FIFO Empty Flag, should be 0, if AF has been initialized successfully by L1Reset
	unsigned zero_4              : 1; // format specific

	// Optical Receiver Status options:
	// {receiver_status_dv, receiver_status_er} = {Receive Data Valid, Receive Error}
	// {0,0} - Receive Idle Character;
	// {0,1} - Receive Carrier Extend;
	// {1,0} - Receive Normal Data Character <- expect to have;
	// {1,1} - Receive Error Propagation;

	// Other data members logically belong to ME Block record,
	//  but physically are located in Data Block Header, which implementation is:
	friend class CSCSPRecord;
	friend class CSCTFPacker;
	// Let this class set following data memebers:
	unsigned int tbin_;           // time bin, that this ME block belongs to in global SP record
	unsigned int valid_pattern;   // LCT valid bit
	unsigned int sync_error;      // LCT synchronization error bit
	unsigned int sync_modified;   // LCT modified synchronization error bit
	unsigned int alignment_fifo;  // AF error
	unsigned int bxBit;           // monitors the ALCT/TMB/MPC timing
	unsigned int spInput_;        // Input SP link, this LCT come through [1..15] (as SP sees it)

public:
	bool check(void) const throw() { return zero_1!=0||zero_2!=0||zero_3!=0||zero_4!=0; }

	unsigned int quality(void) const throw() { return quality_; }
	unsigned int BXN    (void) const throw() { return me_bxn;   }
	unsigned int bx0    (void) const throw() { return bx0_;     }
	unsigned int bc0    (void) const throw() { return bc0_;     }

	unsigned int spInput(void) const throw() { return spInput_; }
	unsigned int link   (void) const throw() { return link_id;  }
	unsigned int mpc    (void) const throw() { return mpc_id;   }
	unsigned int csc    (void) const throw() { return csc_id;   }

	unsigned int l_r    (void) const throw() { return left_right;        }
	unsigned int wireGroup(void) const throw() { return wire_group_id;   }
	unsigned int strip  (void) const throw() { return clct_pattern_id;      }
	unsigned int pattern(void) const throw() { return clct_pattern_number;  }

	enum AF { EMPTY=1, FULL=2 };
	unsigned int aligment_fifo(void) const throw() { return (aligment_fifo_full<<1)|aligment_fifo_empty; }

	enum RS { IDLE_CHARs=0, CARRIER_EXTEND=1, NORMAL_DATA=2, ERROR_PROP=3 };
	unsigned int receiver_status_frame1(void) const throw() { return (receiver_status_dv1<<1)|receiver_status_er1; }
	unsigned int receiver_status_frame2(void) const throw() { return (receiver_status_dv2<<1)|receiver_status_er2; }

	unsigned int errCnt(void) const throw() { return err_prop_cnt; }

	unsigned int tbin(void) const throw() { return tbin_;         }
	unsigned int vp  (void) const throw() { return valid_pattern; }
	unsigned int se  (void) const throw() { return sync_error;    }
	unsigned int sm  (void) const throw() { return sync_modified; }
	unsigned int af  (void) const throw() { return alignment_fifo;}
	unsigned int timingError(void) const throw() { return bxBit; }

	bool unpack(const unsigned short *&buf) throw() { std::memcpy((void*)this,buf,4*sizeof(short)); buf+=4; return check(); }

	CSCSP_MEblock(void){}
};

#endif
