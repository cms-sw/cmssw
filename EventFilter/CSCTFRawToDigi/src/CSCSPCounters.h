#ifndef CSCSPCounters_h
#define CSCSPCounters_h

class CSCSPCounters {
private:
	// Block of counters
	/////// word 1 ///////
	unsigned track_counter_low  : 12; //
	unsigned zero_1             : 4;  //
	/////// word 2 ///////
	unsigned track_counter_high : 12; //
	unsigned zero_2             : 4;  //
	/////// word 3 ///////
	unsigned orbit_counter_low  : 12; //
	unsigned zero_3             : 4;  //
	/////// word 4 ///////
	unsigned orbit_counter_high : 12; //
	unsigned zero_4             : 4;  //

public:
	bool check(void) const throw() {
		return zero_1 !=0 || zero_2 !=0 || zero_3 !=0 || zero_4 !=0 ;
	}

	bool unpack(const unsigned short *&buf) throw() { memcpy((void*)this,buf,4*sizeof(short)); buf+=4; return check(); }

	int track_counter(void) const throw() { return (track_counter_high<<12) | track_counter_low; }
	int orbit_counter(void) const throw() { return (orbit_counter_high<<12) | orbit_counter_low; }

	CSCSPCounters(void){}
};

#endif
