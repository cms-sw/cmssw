// Name: ptHatFilter.cc
// Description:  Filter events in a range of Monte Carlo ptHat.
// Author: R. Harris
// Date:  28 - October - 2008
#include "RecoJets/JetAnalyzers/interface/ptHatFilter.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TFile.h>
#include <cmath>
using namespace edm;
using namespace reco;
using namespace std;
////////////////////////////////////////////////////////////////////////////////////////
ptHatFilter::ptHatFilter(edm::ParameterSet const& cfg)
{
  ptHatLowerCut = cfg.getParameter<double> ("ptHatLowerCut");
  ptHatUpperCut  = cfg.getParameter<double> ("ptHatUpperCut");
}
////////////////////////////////////////////////////////////////////////////////////////
void ptHatFilter::beginJob() 
{
 totalEvents=0;
 acceptedEvents=0;
}

///////////////////////////////////////////////////////////////////////////////
ptHatFilter::~ptHatFilter() {
}

////////////////////////////////////////////////////////////////////////////////////////
bool ptHatFilter::filter(edm::Event& evt, edm::EventSetup const& iSetup) 
{
  
    bool result = false;
    totalEvents++;
    edm::Handle< double > genEventScale;
    evt.getByLabel("genEventScale", genEventScale );
    double pt_hat = *genEventScale;
    if(pt_hat>ptHatLowerCut && pt_hat<ptHatUpperCut)
    {
      acceptedEvents++;
      result = true;
    } 
    return result;
}
////////////////////////////////////////////////////////////////////////////////////////
void ptHatFilter::endJob() 
{
 std::cout << "Total Events = "   << totalEvents << std::endl;
 std::cout << "Accepted Events = "   << acceptedEvents << std::endl;
}
/////////// Register Modules ////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ptHatFilter);
