#ifndef GeneratorInterface_PartonShowerVeto_JetMatchingMadggraph_h
#define GeneratorInterface_PartonShowerVeto_JetMatchingMadgraph_h


#include "GeneratorInterface/PartonShowerVeto/interface/JetMatching.h"

namespace gen
{

class JetMatchingMadgraph : public JetMatching {
    public:
	JetMatchingMadgraph(const edm::ParameterSet &params);
	~JetMatchingMadgraph();

    private:
	void init(const lhef::LHERunInfo* runInfo);
	void beforeHadronisation(const lhef::LHEEvent* event);
	void beforeHadronisationExec();

	int match(const HepMC::GenEvent *partonLevel,
	          const HepMC::GenEvent *finalState,
	          bool showeredFinalState);

	std::set<std::string> capabilities() const;

	template<typename T>
	static T parseParameter(const std::string &value);
	template<typename T>
	T getParameter(const std::string &var, const T &defValue = T()) const;

	std::map<std::string, std::string>	mgParams;

	bool					runInitialized;
	bool					eventInitialized;
	bool					soup;
	bool					exclusive;
};

} // namespace gen


#endif // GeneratorCommon_PartonShowerVeto_JetMatchingMadgraph_h
