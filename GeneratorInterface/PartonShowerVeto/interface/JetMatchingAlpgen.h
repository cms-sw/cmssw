#ifndef GeneratorInterface_PartonShowerVeto_JetMatchingMadggraph_h
#define GeneratorInterface_PartonShowerVeto_JetMatchingMadgraph_h


#include "GeneratorInterface/PartonShowerVeto/interface/JetMatching.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenHeader.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenCommonBlocks.h"

namespace gen
{

class JetMatchingAlpgen : public JetMatching {
    public:
	JetMatchingAlpgen(const edm::ParameterSet &params);
	~JetMatchingAlpgen();

    private:
	void init(const lhef::LHERunInfo* runInfo);
	void beforeHadronisation(const lhef::LHEEvent* event);
	
	int match( const lhef::LHEEvent* partonLevel, const std::vector<fastjet::PseudoJet>* jetInput );
	virtual double getJetEtaMax() const { return 0.; }
/*
	int match(const HepMC::GenEvent* partonLevel,
		  const HepMC::GenEvent* finalState,
		  bool showeredFinalState);
*/
	std::set<std::string> capabilities() const;

	bool applyMatching;
	bool runInitialized;
	bool eventInitialized;

	AlpgenHeader header;
};

} // namespace gen


#endif // GeneratorInterface_PartonShowerVeto_JetMatchingAlpgen_h
