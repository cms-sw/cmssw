#ifndef GeneratorInterface_LHEInterface_JetMatching_h
#define GeneratorInterface_LHEInterface_JetMatching_h

#include <memory>
#include <vector>
#include <string>
#include <set>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/SimpleVector.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace lhef {

class LHERunInfo;
class LHEEvent;
class JetInput;
class JetClustering;

class JetMatching {
    public:
	JetMatching(const edm::ParameterSet &params);
	virtual ~JetMatching();

	struct JetPartonMatch {
		JetPartonMatch(const HepMC::FourVector	&parton,
		               const HepMC::FourVector	&jet,
		               double			delta,
		               int			pdgId) :
			parton(parton), jet(jet),
			delta(delta), pdgId(pdgId) {}

		JetPartonMatch(const HepMC::FourVector	&parton,
		               int			pdgId) :
			parton(parton),	delta(-1.0), pdgId(pdgId) {}

		JetPartonMatch(const HepMC::FourVector &jet) :
			jet(jet), delta(-1.0), pdgId(0) {}

		inline bool isMatch() const { return delta >= 0 && pdgId; }
		inline bool hasParton() const { return pdgId; }
		inline bool hasJet() const { return delta >= 0 || !pdgId; }

		HepMC::FourVector	parton;
		HepMC::FourVector	jet;
		double			delta;
		int			pdgId;
	};

	virtual void init(const boost::shared_ptr<LHERunInfo> &runInfo);
	virtual void beforeHadronisation(
				const boost::shared_ptr<LHEEvent> &event);

	virtual double match(const HepMC::GenEvent *partonLevel,
	                     const HepMC::GenEvent *finalState,
	                     bool showeredFinalState = false) = 0;

	virtual std::set<std::string> capabilities() const;

	const std::vector<JetPartonMatch> &getMatchSummary() const
	{ return matchSummary; }

	static std::auto_ptr<JetMatching> create(
					const edm::ParameterSet &params);

	typedef edmplugin::PluginFactory<JetMatching*(
					const edm::ParameterSet &)> Factory;

    protected:
	std::vector<JetPartonMatch>	matchSummary;
};

} // namespace lhef

#define DEFINE_LHE_JETMATCHING_PLUGIN(T) \
	DEFINE_EDM_PLUGIN(lhef::JetMatching::Factory, T, #T)

#endif // GeneratorCommon_LHEInterface_JetMatching_h
