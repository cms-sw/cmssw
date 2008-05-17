#ifndef GeneratorInterface_LHEInterface_Hadronisation_h
#define GeneratorInterface_LHEInterface_Hadronisation_h

#include <memory>

#include <boost/shared_ptr.hpp>
#include <sigc++/signal.h>

#include <HepMC/GenEvent.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace lhef {

class LHEEvent;
class LHERunInfo;

class Hadronisation {
    public:
	Hadronisation(const edm::ParameterSet &params);
	virtual ~Hadronisation();

	void init();
	bool setEvent(const boost::shared_ptr<LHEEvent> &event);
	void clear();

	std::auto_ptr<HepMC::GenEvent> hadronize();

	inline sigc::signal<bool, const boost::shared_ptr<HepMC::GenEvent>&>&
					onShoweredEvent() { return sigShower; }
	inline bool wantsShoweredEvent() const { return !sigShower.empty(); }

	static std::auto_ptr<Hadronisation> create(
					const edm::ParameterSet &params);

	typedef edmplugin::PluginFactory<Hadronisation*(
					const edm::ParameterSet &)> Factory;

    protected:
	bool showeredEvent(const boost::shared_ptr<HepMC::GenEvent> &event);

	inline const boost::shared_ptr<LHEEvent> &getRawEvent() const
	{ return rawEvent; }

	virtual void doInit() = 0;
	virtual std::auto_ptr<HepMC::GenEvent> doHadronisation() = 0;
	virtual void newRunInfo(const boost::shared_ptr<LHERunInfo> &runInfo);

    private:
	sigc::signal<bool, const boost::shared_ptr<HepMC::GenEvent>&>	sigShower;
	boost::shared_ptr<LHEEvent>					rawEvent;
};

} // namespace lhef

#define DEFINE_LHE_HADRONISATION_PLUGIN(T) \
	DEFINE_EDM_PLUGIN(lhef::Hadronisation::Factory, T, #T)

#endif // GeneratorRunInfo_LHEInterface_Hadronisation_h
