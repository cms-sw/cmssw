#ifndef L1TObjects_L1MuCSCTFConfiguration_h
#define L1TObjects_L1MuCSCTFConfiguration_h

#include <string>
<<<<<<< L1MuCSCTFConfiguration.h
#include <FWCore/ParameterSet/interface/ParameterSet.h>
=======
#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCTFConfigProducer;
>>>>>>> 1.3

class L1MuCSCTFConfiguration {
private:
	std::string registers[12];

public:
	const std::string* configAsText(void) const throw() {
		return registers;
	}

<<<<<<< L1MuCSCTFConfiguration.h
	edm::ParameterSet parameters(int sp) const ;

=======
	edm::ParameterSet parse(void) const ;

>>>>>>> 1.3
	L1MuCSCTFConfiguration& operator=(const L1MuCSCTFConfiguration& conf){
		for(int sp=0;sp<12;sp++) registers[sp] = conf.registers[sp];
		return *this;
	}

	L1MuCSCTFConfiguration(void){}
	L1MuCSCTFConfiguration(std::string regs[12]){ for(int sp=0;sp<12;sp++) registers[sp]=regs[sp]; }
	L1MuCSCTFConfiguration(const L1MuCSCTFConfiguration& conf){
		for(int sp=0;sp<12;sp++) registers[sp] = conf.registers[sp];
	}
	~L1MuCSCTFConfiguration(void){}
};

#endif
