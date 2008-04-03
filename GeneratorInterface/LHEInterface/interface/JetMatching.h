#ifndef GeneratorInterface_LHEInterface_JetMatching_h
#define GeneratorInterface_LHEInterface_JetMatching_h

#include <memory>
#include <vector>

#include <HepMC/GenEvent.h>
#include <HepMC/SimpleVector.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lhef {

class JetInput;
class JetClustering;

class JetMatching {
    public:
	~JetMatching();

	double match(const HepMC::GenEvent *partonLevel,
	             const HepMC::GenEvent *finalState);

	static std::auto_ptr<JetMatching> create(
					const edm::ParameterSet &params);

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

	const std::vector<JetPartonMatch> &getMatchSummary() const
	{ return matchSummary; }

    private:
	enum MatchMode {
		kExclusive = 0,
		kInclusive
	};

	JetMatching();
	JetMatching(const edm::ParameterSet &params);

	const double			maxDeltaR;
	const double			minJetPt;
	MatchMode			matchMode;

	std::auto_ptr<JetInput>		partonInput;
	std::auto_ptr<JetInput>		jetInput;
	std::auto_ptr<JetClustering>	jetClustering;

	std::vector<JetPartonMatch>	matchSummary;

};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_JetMatching_h
