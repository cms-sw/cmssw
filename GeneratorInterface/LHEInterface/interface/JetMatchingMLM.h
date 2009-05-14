#ifndef GeneratorInterface_LHEInterface_JetMatchingMLM_h
#define GeneratorInterface_LHEInterface_JetMatchingMLM_h

#include <memory>
#include <vector>

#include <HepMC/GenEvent.h>
#include <HepMC/SimpleVector.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"

namespace lhef {

class JetInput;
class JetClustering;

class JetMatchingMLM : public JetMatching {
    public:
	JetMatchingMLM(const edm::ParameterSet &params);
	~JetMatchingMLM();

    private:
	std::set<std::string> capabilities() const;

	double match(const HepMC::GenEvent *partonLevel,
	             const HepMC::GenEvent *finalState,
	             bool showeredFinalState);

	enum MatchMode {
		kExclusive = 0,
		kInclusive
	};

	const double			maxDeltaR;
	const double			minJetPt;
	double				maxEta;
	double				matchPtFraction;
	bool				useEt;
	MatchMode			matchMode;

	std::auto_ptr<JetInput>		partonInput;
	std::auto_ptr<JetInput>		jetInput;
	std::auto_ptr<JetClustering>	jetClustering;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_JetMatchingMLM_h
