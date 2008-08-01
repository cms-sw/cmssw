#ifndef L1TObjects_L1MuCSCDTLut_h
#define L1TObjects_L1MuCSCDTLut_h

#include <string.h>

class CSCTFConfigProducer;

class L1MuCSCDTLut {
private:
	unsigned short dt_lut[2][1<<19];
	friend class CSCTFConfigProducer;

public:
	unsigned short dt1(unsigned long addr) const throw() {
		if( addr<(1<<19) ) return dt_lut[0][addr];
		else return 0;
	}

	unsigned short dt2(unsigned long addr) const throw() {
		if( addr<(1<<19) ) return dt_lut[1][addr];
		else return 0;
	}

	const unsigned short* lut(unsigned short dt) const throw() { return dt_lut[dt]; }

	L1MuCSCDTLut& operator=(const L1MuCSCDTLut& lut){ memcpy(dt_lut,lut.dt_lut,sizeof(dt_lut)); return *this; }

	L1MuCSCDTLut(void){ bzero(dt_lut,sizeof(dt_lut)); }
	L1MuCSCDTLut(const L1MuCSCDTLut& lut){ memcpy(dt_lut,lut.dt_lut,sizeof(dt_lut)); }
	~L1MuCSCDTLut(void){}
};

#endif
