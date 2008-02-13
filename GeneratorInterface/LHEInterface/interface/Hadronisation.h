#ifndef GeneratorInterface_LHEInterface_Hadronisation_h
#define GeneratorInterface_LHEInterface_Hadronisation_h

#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lhef {

class LHEEvent;
class LHECommon;

class Hadronisation {
    public:
	Hadronisation(const edm::ParameterSet &params);
	virtual ~Hadronisation();

	void setEvent(const boost::shared_ptr<LHEEvent> &event);
	void clear();

	virtual std::auto_ptr<HepMC::GenEvent> hadronize() = 0;

	static Hadronisation *create(const edm::ParameterSet &params);

    protected:
	virtual void newCommon(const boost::shared_ptr<LHECommon> &common);

	inline const boost::shared_ptr<LHEEvent> &getRawEvent() const
	{ return rawEvent; }

    private:
	boost::shared_ptr<LHEEvent>		rawEvent;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_Hadronisation_h
