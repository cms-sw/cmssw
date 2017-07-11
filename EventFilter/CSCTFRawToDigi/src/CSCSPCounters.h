#ifndef CSCSPCounters_h
#define CSCSPCounters_h

class CSCSPCounters {
private:
	// Block of counters
	/////// word 1 ///////
	unsigned track_counter_low  : 15; //
	unsigned zero_1             : 1;  //
	/////// word 2 ///////
	unsigned track_counter_high : 15; //
	unsigned zero_2             : 1;  //
	/////// word 3 ///////
	unsigned orbit_counter_low  : 15; //
	unsigned zero_3             : 1;  //
	/////// word 4 ///////
	unsigned orbit_counter_high : 15; //
	unsigned zero_4             : 1;  //

	friend class CSCTFPacker;

public:
	bool check() const throw() {
		return zero_1 !=0 || zero_2 !=0 || zero_3 !=0 || zero_4 !=0 ;
	}

	bool unpack(const unsigned short *&buf) throw() { memcpy((void*)this,buf,4*sizeof(short)); buf+=4; return check(); }

	int track_counter() const throw() { return (track_counter_high<<15) | track_counter_low; }
	int orbit_counter() const throw() { return (orbit_counter_high<<15) | orbit_counter_low; }

	CSCSPCounters(){}
};

#endif
