#include "GeneratorInterface/Core/interface/GenFilterEfficiencyAnalyzer.h"

GenFilterEfficiencyAnalyzer::GenFilterEfficiencyAnalyzer(const edm::ParameterSet& pset):
  nTota_(0),nPass_(0),
  genFilterInfoTag_(pset.getParameter<edm::InputTag>("genFilterInfoTag"))
{
}

GenFilterEfficiencyAnalyzer::~GenFilterEfficiencyAnalyzer()
{
}

void
GenFilterEfficiencyAnalyzer::analyze(const edm::Event&, const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------

void
GenFilterEfficiencyAnalyzer::endLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) {

  edm::Handle<GenFilterInfo> genFilter;
  iLumi.getByLabel(genFilterInfoTag_, genFilter);

  std::cout << "Lumi section " << iLumi.id() << std::endl;
  nTota_ += genFilter->numEventsTried();
  nPass_ += genFilter->numEventsPassed();
  std::cout << "N total = " << genFilter->numEventsTried() << " N passed = " << genFilter->numEventsPassed() << std::endl;
  std::cout << "Generator filter efficiency = " << genFilter->filterEfficiency() << " +- " << genFilter->filterEfficiencyError() << std::endl;
  
}

void
GenFilterEfficiencyAnalyzer::endJob() {

  double eff = ( nTota_ > 0 ? (double)nPass_/(double)nTota_ : 1. ) ;
  double err = ( nTota_ > 0 ? std::sqrt((double)nPass_*(1.-(double)nPass_/(double)nTota_))/(double)nTota_ : 1. ); 
  std::cout << "Total events = " << nTota_ << " Passed events = " << nPass_ << std::endl;
  std::cout << "Filter efficiency = " << eff << " +- " << err << std::endl;

}
