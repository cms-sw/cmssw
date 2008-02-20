#ifndef GeneratorInterface_LHEInterface_JetMatching_h
#define GeneratorInterface_LHEInterface_JetMatching_h

#include <memory>
#include <vector>

#include <HepMC/GenEvent.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace lhef {

class JetInput;
class JetClustering;

class JetMatching {
    public:
	~JetMatching();

	double match(const HepMC::GenEvent *partonLevel,
	             const HepMC::GenEvent *finalState) const;

	static std::auto_ptr<JetMatching> create(
					const edm::ParameterSet &params);

    private:
	enum MatchMode {
		kExclusive = 0,
		kInclusive
	};

	JetMatching();
	JetMatching(const edm::ParameterSet &params);

	std::auto_ptr<JetInput>		partonInput;
	std::auto_ptr<JetInput>		jetInput;
	std::auto_ptr<JetClustering>	jetClustering;

	double				maxDeltaR;
	MatchMode			matchMode;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_JetMatching_h
