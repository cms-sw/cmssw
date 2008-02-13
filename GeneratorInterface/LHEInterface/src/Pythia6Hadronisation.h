#ifndef GeneratorInterface_LHEInterface_Pythia6Hadronisation_h
#define GeneratorInterface_LHEInterface_Pythia6Hadronisation_h

#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/IO_HEPEVT.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lhef {

class LHECommon;

class Pythia6Hadronisation : public Hadronisation {
    public:
	Pythia6Hadronisation(const edm::ParameterSet &params);
	~Pythia6Hadronisation();

	struct FortranCallback;

    protected:
	friend struct FortranCallback;

	void fillHeader();
	void fillEvent();
	bool veto();

    private:
	std::auto_ptr<HepMC::GenEvent> hadronize();
	void newCommon(const boost::shared_ptr<LHECommon> &common);

	const int		pythiaPylistVerbosity;
	int			maxEventsToPrint;

	HepMC::IO_HEPEVT	conv;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_Pythia6Hadronisation_h
