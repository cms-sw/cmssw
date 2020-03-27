// F. Cossutti
// $Revision://

// analyzer of a summary information product on filter efficiency for a user specified path
// meant for the generator filter efficiency calculation

// system include files
#include <memory>
#include <iostream>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
//
// class declaration
//

class GenFilterEfficiencyAnalyzer : public edm::EDAnalyzer {
public:
  explicit GenFilterEfficiencyAnalyzer(const edm::ParameterSet&);
  ~GenFilterEfficiencyAnalyzer() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endJob() override;

  edm::EDGetTokenT<GenFilterInfo> genFilterInfoToken_;
  GenFilterInfo totalGenFilterInfo_;

  // ----------member data ---------------------------
};

GenFilterEfficiencyAnalyzer::GenFilterEfficiencyAnalyzer(const edm::ParameterSet& pset)
    : genFilterInfoToken_(consumes<GenFilterInfo, edm::InLumi>(pset.getParameter<edm::InputTag>("genFilterInfoTag"))),
      totalGenFilterInfo_(0, 0, 0, 0, 0., 0., 0., 0.) {}

GenFilterEfficiencyAnalyzer::~GenFilterEfficiencyAnalyzer() {}

void GenFilterEfficiencyAnalyzer::analyze(const edm::Event&, const edm::EventSetup&) {}

// ------------ method called once each job just after ending the event loop  ------------

void GenFilterEfficiencyAnalyzer::endLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {
  edm::Handle<GenFilterInfo> genFilter;
  iLumi.getByToken(genFilterInfoToken_, genFilter);

  std::cout << "Lumi section " << iLumi.id() << std::endl;

  std::cout << "N total = " << genFilter->sumWeights() << " N passed = " << genFilter->sumPassWeights()
            << " N failed = " << genFilter->sumFailWeights() << std::endl;
  std::cout << "Generator filter efficiency = " << genFilter->filterEfficiency(-1) << " +- "
            << genFilter->filterEfficiencyError(-1) << std::endl;
  totalGenFilterInfo_.mergeProduct(*genFilter);
}

void GenFilterEfficiencyAnalyzer::endJob() {
  std::cout << "Total events = " << totalGenFilterInfo_.sumWeights()
            << " Passed events = " << totalGenFilterInfo_.sumPassWeights() << std::endl;
  std::cout << "Filter efficiency = " << totalGenFilterInfo_.filterEfficiency(-1) << " +- "
            << totalGenFilterInfo_.filterEfficiencyError(-1) << std::endl;
}

DEFINE_FWK_MODULE(GenFilterEfficiencyAnalyzer);
