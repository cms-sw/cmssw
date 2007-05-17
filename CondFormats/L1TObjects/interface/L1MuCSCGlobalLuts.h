#ifndef L1TObjects_L1MuCSCGlobalLuts_h
#define L1TObjects_L1MuCSCGlobalLuts_h

#include <string.h>

class CSCTFConfigProducer;

class L1MuCSCGlobalLuts {
private:
	unsigned short global_phi_lut[5][1<<19];
	unsigned short global_eta_lut[5][1<<19];
	friend class CSCTFConfigProducer;

public:
	unsigned short phi(unsigned short mpc, unsigned long addr) const throw() {
		if( addr<(1<<19) ) return global_phi_lut[mpc][addr];
		else return 0;
	}

	unsigned short eta(unsigned short mpc, unsigned long addr) const throw() {
		if( addr<(1<<19) ) return global_eta_lut[mpc][addr];
		else return 0;
	}

	const unsigned short* phi_lut(unsigned short mpc) const throw() { return global_phi_lut[mpc]; }
	const unsigned short* eta_lut(unsigned short mpc) const throw() { return global_eta_lut[mpc]; }

	L1MuCSCGlobalLuts& operator=(const L1MuCSCGlobalLuts& lut){
		memcpy(global_phi_lut,lut.global_phi_lut,sizeof(global_phi_lut));
		memcpy(global_eta_lut,lut.global_eta_lut,sizeof(global_eta_lut));
		return *this;
	}
	L1MuCSCGlobalLuts(void){
		bzero(global_phi_lut,sizeof(global_phi_lut));
		bzero(global_eta_lut,sizeof(global_eta_lut));
	}
	L1MuCSCGlobalLuts(const L1MuCSCGlobalLuts& lut){
		memcpy(global_phi_lut,lut.global_phi_lut,sizeof(global_phi_lut));
		memcpy(global_eta_lut,lut.global_eta_lut,sizeof(global_eta_lut));
	}
	~L1MuCSCGlobalLuts(void){}
};

#endif
