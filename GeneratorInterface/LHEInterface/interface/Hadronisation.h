#ifndef GeneratorInterface_LHEInterface_Hadronisation_h
#define GeneratorInterface_LHEInterface_Hadronisation_h

#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace lhef {

class LHEEvent;
class LHECommon;

class Hadronisation {
    public:
	Hadronisation(const edm::ParameterSet &params);
	virtual ~Hadronisation();

	bool setEvent(const boost::shared_ptr<LHEEvent> &event);
	void clear();

	std::auto_ptr<HepMC::GenEvent> hadronize();

	static std::auto_ptr<Hadronisation> create(
					const edm::ParameterSet &params);

	typedef edmplugin::PluginFactory<Hadronisation*(
					const edm::ParameterSet &)> Factory;

    protected:
	virtual std::auto_ptr<HepMC::GenEvent> doHadronisation() = 0;

	virtual void newCommon(const boost::shared_ptr<LHECommon> &common);

	inline const boost::shared_ptr<LHEEvent> &getRawEvent() const
	{ return rawEvent; }

    private:
	boost::shared_ptr<LHEEvent>		rawEvent;
};

} // namespace lhef

#define DEFINE_LHE_HADRONISATION_PLUGIN(T) \
	DEFINE_EDM_PLUGIN(lhef::Hadronisation::Factory, T, #T)

#endif // GeneratorCommon_LHEInterface_Hadronisation_h
