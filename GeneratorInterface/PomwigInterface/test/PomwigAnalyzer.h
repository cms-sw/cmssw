#ifndef POMWIG_ANALYZER
#define POMWIG_ANALYZER

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HepMC/WeightContainer.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TH1D.h"
#include "TFile.h"

//
// class declaration
//

class PomwigAnalyzer : public edm::EDAnalyzer {
   public:
      explicit PomwigAnalyzer(const edm::ParameterSet&);
      ~PomwigAnalyzer() override;


   private:
      //virtual void beginJob(const edm::EventSetup&) ;
      void beginJob() override;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;

      // ----------member data ---------------------------
      
  std::string outputFilename;
  TH1D* hist_t;
  TH1D* hist_xigen;
  edm::InputTag hepMCProductTag_;
};

#endif
