#ifndef POMWIG_ANALYZER
#define POMWIG_ANALYZER

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
// class declaration
//

class PomwigAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit PomwigAnalyzer(const edm::ParameterSet&);
  ~PomwigAnalyzer() override = default;

private:
  //virtual void beginJob(const edm::EventSetup&) ;
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;
  const std::string outputFilename;
  TH1D* hist_t;
  TH1D* hist_xigen;
};

#endif
