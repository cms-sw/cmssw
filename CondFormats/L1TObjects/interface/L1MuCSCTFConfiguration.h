#ifndef L1TObjects_L1MuCSCTFConfiguration_h
#define L1TObjects_L1MuCSCTFConfiguration_h

#include <string>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCTFConfigProducer;

class L1MuCSCTFConfiguration {
private:
	std::string parametersAsText;
	friend class CSCTFConfigProducer;

public:
	const std::string& parameters(void) const throw() {
		return parametersAsText;
	}

	edm::ParameterSet parse(void) const ;

	L1MuCSCTFConfiguration& operator=(const L1MuCSCTFConfiguration& conf){
		parametersAsText = conf.parametersAsText;
		return *this;
	}

	L1MuCSCTFConfiguration(void){}
	L1MuCSCTFConfiguration(const L1MuCSCTFConfiguration& conf){
		parametersAsText = conf.parametersAsText;
	}
	~L1MuCSCTFConfiguration(void){}
};

#endif
