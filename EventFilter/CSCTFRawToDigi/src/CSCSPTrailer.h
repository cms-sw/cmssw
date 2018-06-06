#ifndef CSCSPTrailer_h
#define CSCSPTrailer_h

#include <cstring>

class CSCSPTrailer {
private:
	/////// word 1 ///////
	unsigned l1a_           : 8;
	unsigned word_count_low : 4;
	unsigned trailer_mark_1 : 4;  // constant, should be 1111 = 0xF
	/////// word 2 ///////
	unsigned trailer_mark_2 : 4;  // constant, should be 1111 = 0xF
	unsigned trailer_mark_3 : 3;  // constant, should be  111 = 0x7
	unsigned l1a_fifo_full_ : 1;
	unsigned word_count_high: 4;
	unsigned trailer_mark_4 : 4;  // constant, should be 1111 = 0xF
	/////// word 3 ///////
	unsigned month_            : 4;
	unsigned year_             : 4;
	unsigned bb_               : 1; // SP readout configuration year base (0 / 16)
	unsigned spare_1           : 1;
	unsigned spare_2           : 1;
	unsigned zero_1            : 1;
	unsigned trailer_mark_5    : 4;  // constant, should be 1111 = 0xF
	/////// word 4 ///////
	unsigned core_configuraton : 12;
	unsigned trailer_mark_6    : 4;  // constant, should be 1111 = 0xF
	/////// word 5 ///////
	unsigned day_              : 5;
	unsigned zero_2            : 7;
	unsigned trailer_mark_7    : 4;  // constant, should be 1110 = 0xE
	/////// word 6 ///////
	unsigned board_id_         : 12;
	unsigned trailer_mark_8    : 4;  // constant, should be 1110 = 0xE
	/////// word 7 ///////
	unsigned crc_low           : 11;
	unsigned crc_low_parity    : 1;
	unsigned trailer_mark_9    : 4;  // constant, should be 1110 = 0xE
	/////// word 8 ///////
	unsigned crc_high          : 11;
	unsigned crc_high_parity   : 1;
	unsigned trailer_mark_10   : 4;  // constant, should be 1110 = 0xE

	friend class CSCTFPacker;

public:
	bool check(void) const throw() {
		return spare_1!=0 || spare_2!=0 || zero_1!=0 || zero_2!=0 ||
			trailer_mark_1!=0xF || trailer_mark_2!=0xF || trailer_mark_3!=0x7 || trailer_mark_4!=0xF || trailer_mark_5!=0xF || trailer_mark_6!=0xF ||
			trailer_mark_7!=0xE || trailer_mark_8!=0xE || trailer_mark_9!=0xE || trailer_mark_10!=0xE;
	}

	unsigned int l1a_7bits     (void) const throw() { return l1a_; }
	unsigned int l1a_queue_size(void) const throw() { return (word_count_high<<4)|word_count_low; }
	bool         l1a_fifo_full (void) const throw() { return l1a_fifo_full_; }

	unsigned int year          (void) const throw() { return 2000+16*bb_+year_; }
	unsigned int month         (void) const throw() { return month_; }
	unsigned int day           (void) const throw() { return day_;   }
	unsigned int configuration (void) const throw() { return core_configuraton;  }

	//unsigned int slot    (void) const throw() { return  board_id_&0x1F;       }
	//unsigned int sector  (void) const throw() { return (board_id_&0x700)>>8;  }
	//unsigned int endcap  (void) const throw() { return (board_id_&0x800)>>11; }

	unsigned int board_id(void) const throw() { return  board_id_;             }

	unsigned int crc     (void) const throw() { return crc_low|(crc_high<<11); }

	bool unpack(const unsigned short *&buf) throw()  { std::memcpy(this, buf, 8*sizeof(short)); buf+=8; return check(); }

	CSCSPTrailer(void){}
};

#endif
