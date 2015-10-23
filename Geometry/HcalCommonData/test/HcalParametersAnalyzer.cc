#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include <iostream>

class HcalParametersAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalParametersAnalyzer( const edm::ParameterSet& );
  ~HcalParametersAnalyzer( void );
  
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}    
};

HcalParametersAnalyzer::HcalParametersAnalyzer( const edm::ParameterSet& ) {}

HcalParametersAnalyzer::~HcalParametersAnalyzer( void ) {}

void HcalParametersAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup ) {
  edm::ESHandle<HcalParameters> parHandle;
  iSetup.get<HcalParametersRcd>().get( parHandle );
  const HcalParameters* pars ( parHandle.product());

  std::cout << "rHB: ";
  for( const auto& it : pars->rHB ) {
    std::cout << it << ", ";
  }
  std::cout << "\ndrHB: ";
  for( const auto& it : pars->drHB ) {
    std::cout << it << ", ";
  }
  std::cout << "\nzHE: ";
  for( const auto& it : pars->zHE ) {
    std::cout << it << ", ";
  }
  std::cout << "\ndzHE: ";
  for( const auto& it : pars->dzHE ) {
    std::cout << it << ", ";
  }
  std::cout << "\nzHO: ";
  for( const auto& it : pars->zHO ) {
    std::cout << it << ", ";
  }
  std::cout << "\nrHO: ";
  for( const auto& it : pars->rHO ) {
      std::cout << it << ", ";
  }
  std::cout << "\nrhoxHB: ";
  for( const auto& it : pars->rhoxHB ) {
    std::cout << it << ", ";
  }
  std::cout << "\nzxHB: ";
  for( const auto& it : pars->zxHB ) {
    std::cout << it << ", ";
  }
  std::cout << "\ndyHB: ";
  for( const auto& it : pars->dyHB ) {
    std::cout << it << ", ";
  }
  std::cout << "\ndxHB: ";
  for( const auto& it : pars->dxHB ) {
    std::cout << it << ", ";
  }
  std::cout << "\nrhoxHE: ";
  for( const auto& it : pars->rhoxHE ) {
    std::cout << it << ", ";
  }
  std::cout << "\nzxHE: ";
  for( const auto& it : pars->zxHE ) {
    std::cout << it << ", ";
  }
  std::cout << "\ndyHE: ";
  for( const auto& it : pars->dyHE ) {
    std::cout << it << ", ";
  }
  std::cout << "\ndx1HE: ";
  for( const auto& it : pars->dx1HE ) {
    std::cout << it << ", ";
  }
  std::cout << "\ndx2HE: ";
  for( const auto& it : pars->dx2HE ) {
    std::cout << it << ", ";
  }
  std::cout << "\nphioff: ";
  for( const auto& it : pars->phioff ) {
    std::cout << it << ", ";
  }
  std::cout << "\netaTable: ";
  for( const auto& it : pars->etaTable ) {
    std::cout << it << ", ";
  }
  std::cout << "\nrTable: ";
  for( const auto& it : pars->rTable ) {
    std::cout << it << ", ";
  }
  std::cout << "\nphibin: ";
  for( const auto& it : pars->phibin ) {
    std::cout << it << ", ";
  }
  std::cout << "\nphitable: ";
  for( const auto& it : pars->phitable ) {
    std::cout << it << ", ";
  }
  std::cout << "\netaRange: ";
  for( const auto& it : pars->etaRange ) {
    std::cout << it << ", ";
  }
  std::cout << "\ngparHF: ";
  for( const auto& it : pars->gparHF ) {
    std::cout << it << ", ";
  }
  std::cout << "\nLayer0Wt: ";
  for( const auto& it : pars->Layer0Wt ) {
    std::cout << it << ", ";
  }
  std::cout << "\nHBGains: ";
  for( const auto& it : pars->HBGains ) {
    std::cout << it << ", ";
  }
  std::cout << "\nHEGains: ";
  for( const auto& it : pars->HEGains ) {
    std::cout << it << ", ";
  }
  std::cout << "\nHFGains: ";
  for( const auto& it : pars->HFGains ) {
    std::cout << it << ", ";
  }
  std::cout << "\netaTableHF: ";
  for( const auto& it : pars->etaTableHF ) {
    std::cout << it << ", ";
  }
  std::cout << "\nmodHB: ";
  for( const auto& it : pars->modHB ) {
    std::cout << it << ", ";
  }
  std::cout << "\nmodHE: ";
  for( const auto& it : pars->modHE ) {
    std::cout << it << ", ";
  }
  std::cout << "\nmaxDepth: ";
  for( const auto& it : pars->maxDepth ) {
    std::cout << it << ", ";
  }
  std::cout << "\nlayHB: ";
  for( const auto& it : pars->layHB ) {
    std::cout << it << ", ";
  }
  std::cout << "\nlayHE: ";
  for( const auto& it : pars->layHE ) {
    std::cout << it << ", ";
  }
  std::cout << "\netaMin: ";
  for( const auto& it : pars->etaMin ) {
    std::cout << it << ", ";
  }
  std::cout << "\netaMax: ";
  for( const auto& it : pars->etaMax ) {
    std::cout << it << ", ";
  }
  std::cout << "\nnoff: ";
  for( const auto& it :  pars->noff ) {
    std::cout << it << ", ";
  }      
  std::cout << "\nHBShift: ";
  for( const auto& it : pars->HBShift ) {
    std::cout << it << ", ";
  }
  std::cout << "\nHEShift: ";
  for( const auto& it : pars->HEShift ) {
    std::cout << it << ", ";
    }
  std::cout << "\nHFShift: ";
  for( const auto& it : pars->HFShift ) {
    std::cout << it << ", ";
  }
  for( const auto& it : pars->layerGroupEtaSim ) {
    std::cout << "\nlayerGroupEtaSim" << it.layer << ": ";
    for( const auto& iit : it.layerGroup )  {
      std::cout << iit << ", ";
    }
  }
  std::cout << "\netagroup: ";
  for( const auto& it : pars->etagroup )  {
    std::cout << it << ", ";
  }
  std::cout << "\nphigroup: ";
  for( const auto& it : pars->phigroup ) {
    std::cout << it << ", ";
  }
  for( const auto& it : pars->layerGroupEtaRec ) {
    std::cout << "\nlayerGroupEtaRec" << it.layer << ": ";
    for( const auto& iit : it.layerGroup )  {
      std::cout << iit << ", ";
    }
  }
  std::cout << "\ndzVcal: " << pars->dzVcal
	    << "\nTopologyMode: " << pars->topologyMode << std::endl;
}

DEFINE_FWK_MODULE(HcalParametersAnalyzer);
