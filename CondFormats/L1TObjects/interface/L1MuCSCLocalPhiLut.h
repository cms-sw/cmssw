#ifndef L1TObjects_L1MuCSCLocalPhiLut_h
#define L1TObjects_L1MuCSCLocalPhiLut_h

#include <string.h>

class CSCTFConfigProducer;

class L1MuCSCLocalPhiLut {
private:
	unsigned short phi_lut[1<<19];
	friend class CSCTFConfigProducer;

public:
	unsigned short local_phi(unsigned long addr) const throw() {
		if( addr<(1<<19) ) return phi_lut[addr];
		else return 0;
	}

	const unsigned short* lut(void) const throw() { return phi_lut; }

	L1MuCSCLocalPhiLut& operator=(const L1MuCSCLocalPhiLut& lut){ memcpy(phi_lut,lut.phi_lut,sizeof(phi_lut)); return *this; }

	L1MuCSCLocalPhiLut(void){ bzero(phi_lut,sizeof(phi_lut)); }
	L1MuCSCLocalPhiLut(const L1MuCSCLocalPhiLut& lut){ memcpy(phi_lut,lut.phi_lut,sizeof(phi_lut)); }
	~L1MuCSCLocalPhiLut(void){}
};

#endif
