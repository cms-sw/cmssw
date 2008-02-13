#ifndef GeneratorInterface_LHEInterface_LHECommon_h
#define GeneratorInterface_LHEInterface_LHECommon_h

#include <iostream>
#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"

namespace lhef {

class LHECommon {
    public:
	LHECommon(std::istream &in, const std::string &comment);
	~LHECommon();

	const HEPRUP *getHEPRUP() const { return &heprup; } 

    private:
	HEPRUP	heprup;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_LHECommon_h
