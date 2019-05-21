#include "GeneratorInterface/Core/interface/GenFilterEfficiencyAnalyzer.h"
#include <iostream>

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
