#ifndef Z2MU_ANALYZER
#define Z2MU_ANALYZER

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
// class decleration
//

class Z2muAnalyzer : public edm::EDAnalyzer {
   public:
      explicit Z2muAnalyzer(const edm::ParameterSet&);
      ~Z2muAnalyzer();


   private:
      //virtual void beginJob(const edm::EventSetup&);
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      
  std::string outputFilename;
  TH1D* weight_histo;
  TH1D* invmass_histo;
  edm::InputTag hepMCProductTag_;
};

#endif
