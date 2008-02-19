#ifndef GeneratorInterface_LHEInterface_JetInput_h
#define GeneratorInterface_LHEInterface_JetInput_h

#include <vector>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>
#include <HepMC/GenVertex.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lhef {

class JetInput {
    public:
	typedef std::vector<bool>			ParticleBitmap;
	typedef std::vector<const HepMC::GenParticle*>	ParticleVector;

	JetInput();
	JetInput(const edm::ParameterSet &params);
	~JetInput();

	ParticleVector operator () (const HepMC::GenEvent *event) const;

	bool getPartonicFinalState() const { return partonicFinalState; }
	bool getExcludeResonances() const { return excludeResonances; }
	bool getSignalProcessOnly() const { return onlySignalProcess; }
	const std::vector<unsigned int> &getIgnoredParticles() const
	{ return ignoreParticleIDs; }

	void setPartonicFinalState(bool flag = true)
	{ partonicFinalState = flag; }
	void setExcludeResonances(bool flag = true)
	{ excludeResonances = flag; }
	void setSignalProcessOnly(bool flag = true)
	{ onlySignalProcess = flag; }
	void setIgnoredParticles(const std::vector<unsigned int> &particleIDs);

	static bool isParton(int pdgId);
	static bool isHadron(int pdgId);
	static bool isResonance(int pdgId);

	bool isIgnored(int pdgId) const;

	bool hasPartonChildren(ParticleBitmap &invalid,
	                       const ParticleVector &p,
	                       const HepMC::GenParticle *particle) const;
	bool fromSignalVertex(ParticleBitmap &invalid,
	                      const ParticleVector &p,
	                      const HepMC::GenParticle *particle,
	                      const HepMC::GenVertex *signal) const;
	bool fromResonance(ParticleBitmap &invalid,
	                   const ParticleVector &p,
        	           const HepMC::GenParticle *particle,
	                   const HepMC::GenVertex *signal) const;

    private:
	std::vector<unsigned int>	ignoreParticleIDs;
	bool				partonicFinalState;
	bool				excludeResonances;
	bool				onlySignalProcess;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_JetInput_h
