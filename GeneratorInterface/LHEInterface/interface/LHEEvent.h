#ifndef GeneratorInterface_LHEInterface_LHEEvent_h
#define GeneratorInterface_LHEInterface_LHEEvent_h

#include <iostream>
#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommon.h"

namespace lhef {

class LHEEvent {
    public:
	LHEEvent(const boost::shared_ptr<LHECommon> &common,
	         std::istream &in);
	~LHEEvent();

	const boost::shared_ptr<LHECommon> &getCommon() const { return common; }
	const HEPEUP *getHEPEUP() const { return &hepeup; }
	const HEPRUP *getHEPRUP() const { return common->getHEPRUP(); }

    private:
	const boost::shared_ptr<LHECommon>	common;

	HEPEUP					hepeup;
};

} // namespace lhef

#endif // GeneratorEvent_LHEInterface_LHEEvent_h
