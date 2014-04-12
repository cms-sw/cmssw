#ifndef GeneratorInterface_PartonShowerVeto_JetMatchingMadgraph_h
#define GeneratorInterface_PartonShowerVeto_JetMatchingMadgraph_h

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatching.h"

namespace gen
{

class JetMatchingMadgraph : public JetMatching {
    public:
	JetMatchingMadgraph(const edm::ParameterSet &params);
	~JetMatchingMadgraph();

    protected:
	virtual void init(const lhef::LHERunInfo* runInfo);
	virtual void beforeHadronisation(const lhef::LHEEvent* event);
	virtual void beforeHadronisationExec();

        virtual int match( const lhef::LHEEvent* partonLevel, const std::vector<fastjet::PseudoJet>* jetInput );
	
	virtual double                  getJetEtaMax() const;
	
/*
	int match(const HepMC::GenEvent *partonLevel,
	          const HepMC::GenEvent *finalState,
	          bool showeredFinalState);
*/
	std::set<std::string> capabilities() const;

	template<typename T>
	static T parseParameter(const std::string &value);
	template<typename T>
	static T getParameter(const std::map<std::string, std::string> &params,
	                      const std::string &var, const T &defValue = T());
	template<typename T>
	T getParameter(const std::string &var, const T &defValue = T()) const;

	template<typename T>
	static void updateOrDie(
			const std::map<std::string, std::string> &params,
			T &param, const std::string &name);

	std::map<std::string, std::string>	mgParams;

	bool					runInitialized;
	bool					eventInitialized;
	bool					soup;
	bool					exclusive;
};

} // namespace gen


#endif // GeneratorInterface_PartonShowerVeto_JetMatchingMadgraph_h
