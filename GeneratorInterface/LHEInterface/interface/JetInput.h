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
	typedef std::vector<const HepMC::GenVertex*>	VertexVector;

	JetInput();
	JetInput(const edm::ParameterSet &params);
	~JetInput();

	ParticleVector operator () (const HepMC::GenEvent *event) const;

	bool getPartonicFinalState() const { return partonicFinalState; }
	bool getHardProcessOnly() const { return onlyHardProcess; }
	bool getTausAndJets() const { return tausAsJets; }
	bool getExcludeResonances() const { return excludeResonances; }
	double getPtMin() const { return ptMin; }
	const std::vector<unsigned int> &getIgnoredParticles() const
	{ return ignoreParticleIDs; }
	const std::vector<unsigned int> &getExcludedResonances() const
	{ return excludedResonances; }
	const std::vector<unsigned int> &getExcludedFromResonances() const
	{ return excludedFromResonances; }

	void setPartonicFinalState(bool flag = true)
	{ partonicFinalState = flag; }
	void setHardProcessOnly(bool flag = true)
	{ onlyHardProcess = flag; }
	void setTausAsJets(bool flag = true) { tausAsJets = flag; }
	void setExcludeResonances(bool flag = true) { excludeResonances = flag; }
	void setPtMin(double ptMin) { this->ptMin = ptMin; }
	void setIgnoredParticles(const std::vector<unsigned int> &particleIDs);
	void setExcludedResonances(const std::vector<unsigned int> &particleIds);
	void setExcludedFromResonances(const std::vector<unsigned int> &particleIds);

	bool isParton(int pdgId) const;
	static bool isHadron(int pdgId);
	bool isResonance(int pdgId) const;
	bool isExcludedFromResonances(int pdgId) const;
	bool isHardProcess(const HepMC::GenVertex *vertex,
	                   const VertexVector &hardProcess) const;
	bool isIgnored(int pdgId) const;

	enum ResonanceState {
		kNo = 0,
		kDirect,
		kIndirect
	};

	bool hasPartonChildren(ParticleBitmap &invalid,
	                       const ParticleVector &p,
	                       const HepMC::GenParticle *particle) const;
	bool fromHardProcess(ParticleBitmap &invalid,
	                     const ParticleVector &p,
	                     const HepMC::GenParticle *particle,
	                     const VertexVector &hardProcess) const;
	bool fromSignalVertex(ParticleBitmap &invalid,
	                      const ParticleVector &p,
	                      const HepMC::GenParticle *particle,
	                      const HepMC::GenVertex *signalVertex) const;
	ResonanceState fromResonance(ParticleBitmap &invalid,
	                             const ParticleVector &p,
        	                     const HepMC::GenParticle *particle,
	                             const HepMC::GenVertex *signalVertex,
	                             const VertexVector &hardProcess) const;

    private:
	int testPartonChildren(ParticleBitmap &invalid,
	                       const ParticleVector &p,
	                       const HepMC::GenVertex *v) const;

	std::vector<unsigned int>	ignoreParticleIDs;
	std::vector<unsigned int>	excludedResonances;
	std::vector<unsigned int>	excludedFromResonances;
	bool				partonicFinalState;
	bool				onlyHardProcess;
	bool				excludeResonances;
	bool				tausAsJets;
	double				ptMin;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_JetInput_h
