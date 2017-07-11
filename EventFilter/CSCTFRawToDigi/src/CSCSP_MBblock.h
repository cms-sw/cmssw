#ifndef CSCSP_MBblock_h
#define CSCSP_MBblock_h

class CSCSP_MBblock {
private:
	/////// word 1 ///////
	unsigned quality_  : 3; // muon quality (0-7),  quality > 0 for valid data;
	unsigned zero_1    : 1; // format specific
	unsigned phi_bend_ : 5; // phi bend angle
	unsigned zero_2    : 3; // format specific
	unsigned flag_     : 1; // if 1 then it is a second muon from previous bunch crossing
	unsigned cal_      : 1; // MBy special mode flag
	unsigned zero_3    : 2; // format specific
	/////// word 2 ///////
	unsigned phi_      : 12;// azimuth coordinate;
	unsigned bxn1_     : 1; // next to the least significant bit of the MB BX number
	unsigned bxn0_     : 1; // least significant bit of the MB BX number
	unsigned bc0_      : 1; // BX zero timing mark
	unsigned zero_4    : 1; // format specific
	/////// word 3 ///////
	unsigned mb_bxn_   : 12;// Stub arrival time picked from a local 12-bit Bunch Counter, that runs at data stream timing
	unsigned spare_1   : 3; // not used
	unsigned zero_5    : 1; // format specific
	/////// word 4 ///////
	unsigned spare_2   : 15;// not used
	unsigned zero_6    : 1; // format specific

	// Other data members logically belong to MB block record,
	//  but physically are located in Data Block Header, which implementation is:
	friend class CSCSPRecord;
	friend class CSCTFPacker;
	// Let this class set following data memebers:
	unsigned int tbin_;           // time bin, that this MB block belongs to in global SP record
	unsigned int valid_quality;   // valid quality
	unsigned int alignment_fifo;  // AF error
	unsigned int bxBit;           // monitors the MB(DT) timing
	unsigned int id_;             // stub id (1-MB1a, 2-MB1d)

public:
	bool check() const throw() { return zero_1!=0||zero_2!=0||zero_3!=0||zero_4!=0||zero_5!=0||zero_6!=0||spare_1!=0||spare_2!=0; }

	unsigned int quality () const throw() { return quality_;  }
	unsigned int phi_bend() const throw() { return phi_bend_; }
	unsigned int flag    () const throw() { return flag_;     }
	unsigned int cal     () const throw() { return cal_;      }

	unsigned int phi() const throw() { return phi_;    }
	unsigned int bxn() const throw() { return (bxn1_<<1)|bxn0_; }
	unsigned int bc0() const throw() { return bc0_;    }
	unsigned int BXN() const throw() { return mb_bxn_; }

	unsigned int id  () const throw() { return id_;           }
	unsigned int tbin() const throw() { return tbin_;         }
	unsigned int vq  () const throw() { return valid_quality; }
	unsigned int af  () const throw() { return alignment_fifo;}
	unsigned int timingError() const throw() { return bxBit; }

	bool unpack(const unsigned short *&buf) throw() { memcpy(this, buf, 4*sizeof(short)); buf+=4; return check(); }

	CSCSP_MBblock(){}
};

#endif
