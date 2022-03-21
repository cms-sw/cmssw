#ifndef Z2MU_ANALYZER
#define Z2MU_ANALYZER

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "HepMC/WeightContainer.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TH1D.h"
#include "TFile.h"

//
// class decleration
//

class Z2muAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit Z2muAnalyzer(const edm::ParameterSet&);
  ~Z2muAnalyzer() override = default;

private:
  //virtual void beginJob(const edm::EventSetup&);
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;
  const std::string outputFilename;
  TH1D* weight_histo;
  TH1D* invmass_histo;
};

#endif
