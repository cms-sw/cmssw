#ifndef GeneratorInterface_PartonShowerVeto_JetMatchingAlpgen_h
#define GeneratorInterface_PartonShowerVeto_JetMatchingAlpgen_h

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatching.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenHeader.h"
#include "GeneratorInterface/AlpgenInterface/interface/AlpgenCommonBlocks.h"

namespace gen {

  class JetMatchingAlpgen : public JetMatching {
  public:
    JetMatchingAlpgen(const edm::ParameterSet& params);
    ~JetMatchingAlpgen() override;

  private:
    void init(const lhef::LHERunInfo* runInfo) override;
    void beforeHadronisation(const lhef::LHEEvent* event) override;

    int match(const lhef::LHEEvent* partonLevel, const std::vector<fastjet::PseudoJet>* jetInput) override;
    double getJetEtaMax() const override { return 0.; }
    /*
	int match(const HepMC::GenEvent* partonLevel,
		  const HepMC::GenEvent* finalState,
		  bool showeredFinalState);
*/
    std::set<std::string> capabilities() const override;

    bool applyMatching;
    bool runInitialized;
    bool eventInitialized;

    AlpgenHeader header;
  };

}  // namespace gen

#endif  // GeneratorInterface_PartonShowerVeto_JetMatchingAlpgen_h
