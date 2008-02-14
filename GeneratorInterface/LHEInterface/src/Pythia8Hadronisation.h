#ifndef GeneratorInterface_LHEInterface_Pythia8Hadronisation_h
#define GeneratorInterface_LHEInterface_Pythia8Hadronisation_h

#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace Pythia8 {
	class Pythia;
}

namespace HepMC {
	class I_Pythia8;
}

namespace lhef {

class LHECommon;
class LHAinitLesHouches;
class LHAevntLesHouches;

class Pythia8Hadronisation : public Hadronisation {
    public:
	Pythia8Hadronisation(const edm::ParameterSet &params);
	~Pythia8Hadronisation();

    private:
	std::auto_ptr<HepMC::GenEvent> hadronize();
	void newCommon(const boost::shared_ptr<LHECommon> &common);

	const int				pythiaPylistVerbosity;
	int					maxEventsToPrint;

	std::auto_ptr<Pythia8::Pythia>		pythia;
	std::auto_ptr<LHAinitLesHouches>	lhaInit;
	std::auto_ptr<LHAevntLesHouches>	lhaEvent;
	std::auto_ptr<HepMC::I_Pythia8>		conv;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_Pythia8Hadronisation_h
