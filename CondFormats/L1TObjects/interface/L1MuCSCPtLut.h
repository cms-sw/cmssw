#ifndef L1TObjects_L1MuCSCPtLut_h
#define L1TObjects_L1MuCSCPtLut_h

#include <string.h>
#include "CondFormats/Calibration/interface/fixedArray.h"

class CSCTFConfigProducer;

class L1MuCSCPtLut {
private:
        fixedArray< unsigned short, 1<<21 > pt_lut ;
	friend class CSCTFConfigProducer;

public:
	unsigned short pt(unsigned long addr) const throw() {
	  if( addr<(1<<21) ) return pt_lut[(unsigned int)addr];
		else return 0;
	}

	const unsigned short* lut(void) const throw() { return pt_lut; }

	L1MuCSCPtLut& operator=(const L1MuCSCPtLut& lut){ memcpy(pt_lut,lut.pt_lut,sizeof(pt_lut)); return *this; }

	L1MuCSCPtLut(void){ bzero(pt_lut,sizeof(pt_lut)); }
	L1MuCSCPtLut(const L1MuCSCPtLut& lut){ memcpy(pt_lut,lut.pt_lut,sizeof(pt_lut)); }
	~L1MuCSCPtLut(void){}
};

#endif
