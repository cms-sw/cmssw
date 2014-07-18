#ifndef gen_PomwigHadronizer_h
#define gen_PomwigHadronizer_h

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Instance.h"

#include <HepMC/IO_HERWIG.h>

#include <string>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen
{
  class PomwigHadronizer : public gen::BaseHadronizer, public gen::Herwig6Instance
  {
    public:
        PomwigHadronizer(const edm::ParameterSet &params);
        ~PomwigHadronizer();

        bool readSettings( int );
	bool initializeForInternalPartons();
        bool initializeForExternalPartons();

	bool declareStableParticles(const std::vector<int> &pdgIds);
	bool declareSpecialSettings( const std::vector<std::string>& ) { return true; }
        void statistics();

        bool generatePartonsAndHadronize();
        bool hadronize();
        bool decay();
        bool residualDecay();
        void finalizeEvent();

        const char *classname() const { return "PomwigHadronizer"; }

    private:

        virtual void doSetRandomEngine(CLHEP::HepRandomEngine* v) override;
        virtual std::vector<std::string> const& doSharedResources() const override { return theSharedResources; }

        void clear();
        bool initializeDPDF();

        static const std::vector<std::string> theSharedResources;

        bool                            needClear;

        gen::ParameterCollector         parameters;
        int                             herwigVerbosity;
        int                             hepmcVerbosity;
        int                             maxEventsToPrint;
        bool                            printCards;

        double                          comEnergy; 
        double                          survivalProbability;
        int                             diffTopology;
        int                             h1fit;

        bool                            useJimmy;
        bool                            doMPInteraction;
        int                             numTrials;

        bool                            doPDGConvert;

        HepMC::IO_HERWIG		conv;
  };
}
#endif
