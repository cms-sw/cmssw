#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/GeometryObjects/interface/PHcalParameters.h"
#include "Geometry/Records/interface/PHcalParametersRcd.h"
#include <iostream>

class HcalParametersAnalyzer : public edm::EDAnalyzer 
{
public:
    explicit HcalParametersAnalyzer( const edm::ParameterSet& );
    ~HcalParametersAnalyzer( void );
    
    virtual void analyze( const edm::Event&, const edm::EventSetup& );
};

HcalParametersAnalyzer::HcalParametersAnalyzer( const edm::ParameterSet& ) 
{}

HcalParametersAnalyzer::~HcalParametersAnalyzer( void )
{}

void
HcalParametersAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
    edm::ESHandle<PHcalParameters> parHandle;
    iSetup.get<PHcalParametersRcd>().get( parHandle );
    const PHcalParameters* pars ( parHandle.product());

    std::cout << "phioff: ";
    for( const auto& it : pars->phioff )
    {
      std::cout << it << ", ";
    }
    std::cout << "\netaTable: ";
    for( const auto& it : pars->etaTable )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nrTable: ";
    for( const auto& it : pars->rTable )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nphibin: ";
    for( const auto& it : pars->phibin )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nphitable: ";
    for( const auto& it : pars->phitable )
    {
      std::cout << it << ", ";
    }
    std::cout << "\netaRange: ";
    for( const auto& it : pars->etaRange )
    {
      std::cout << it << ", ";
    }
    std::cout << "\ngparHF: ";
    for( const auto& it : pars->gparHF )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nnoff: ";
    for( const auto& it :  pars->noff )
    {
      std::cout << it << ", ";
    }      
    std::cout << "\nLayer0Wt: ";
    for( const auto& it : pars->Layer0Wt )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nHBGains: ";
    for( const auto& it : pars->HBGains )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nHEGains: ";
    for( const auto& it : pars->HEGains )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nHFGains: ";
    for( const auto& it : pars->HFGains )
    {
      std::cout << it << ", ";
    }
    std::cout << "\netaMin: ";
    for( const auto& it : pars->etaMin )
    {
      std::cout << it << ", ";
    }
    std::cout << "\netaMax: ";
    for( const auto& it : pars->etaMax )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nHBShift: ";
    for( const auto& it : pars->HBShift )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nHEShift: ";
    for( const auto& it : pars->HEShift )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nHFShift: ";
    for( const auto& it : pars->HFShift )
    {
      std::cout << it << ", ";
    }
    std::cout << "\netagroup: ";
    for( const auto& it : pars->etagroup )
    {
      std::cout << it << ", ";
    }
    std::cout << "\nphigroup: ";
    for( const auto& it : pars->phigroup )
    {
      std::cout << it << ", ";
    }
    for( const auto& it : pars->layerGroupEta )
    {
      std::cout << "\nlayerGroupEta" << it.layer << ": ";
      for( const auto& iit : it.layerGroup )
      {
	std::cout << iit << ", ";
      }
    }
    std::cout << "\nTopologyMode: " << pars->topologyMode << std::endl;
}

DEFINE_FWK_MODULE(HcalParametersAnalyzer);
