#ifndef GeneratorInterface_LHEInterface_JetClustering_h
#define GeneratorInterface_LHEInterface_JetClustering_h

#include <memory>
#include <vector>

#include <HepMC/GenEvent.h>
#include <HepMC/SimpleVector.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lhef {

class JetClustering {
    public:
	JetClustering(const edm::ParameterSet &params);
	~JetClustering();

	std::vector<HepMC::FourVector> run(const HepMC::GenEvent *event) const;

	class Algorithm;

    private:
	std::vector<HepMC::FourVector> cluster(
				const HepMC::GenEvent *event) const;

	std::vector<unsigned int>	ignoreParticleIDs;

	std::auto_ptr<Algorithm>	jetAlgo;

	bool				partonicFinalState;
	bool				excludeResonances;
	bool				onlySignalProcess;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_JetClustering_h
