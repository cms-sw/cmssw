#ifndef L1TObjects_L1MuCSCTFAlignment_h
#define L1TObjects_L1MuCSCTFAlignment_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

class L1MuCSCTFAlignment {
private:
	std::vector<double> coefficients;

public:
	L1MuCSCTFAlignment& operator=(const L1MuCSCTFAlignment& conf){
		coefficients = conf.coefficients;
		return *this;
	}

	const std::vector<double>& operator()() { return coefficients; }

	L1MuCSCTFAlignment(void){}
	L1MuCSCTFAlignment(const std::vector<double>& cff){ coefficients=cff; }
	L1MuCSCTFAlignment(const L1MuCSCTFAlignment& conf){
		coefficients = conf.coefficients;
	}
	~L1MuCSCTFAlignment(void){}

  COND_SERIALIZABLE;
};

#endif
