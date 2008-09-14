#include "RecoJets/JetAnalyzers/interface/myFilter.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace reco;
using namespace std;


myFilter::myFilter(const edm::ParameterSet& cfg) :
  CaloJetAlgorithm( cfg.getParameter<string>( "CaloJetAlgorithm" ) )
{
  _nEvent      = 0;
  _acceptedEvt = 0;
  _passPt      = 0;
  _passEMF     = 0;
}

myFilter::~myFilter() {
}

void myFilter::beginJob(edm::EventSetup const&) {
}

void myFilter::endJob() {

  std::cout << "myFilter: accepted " 
	    << _acceptedEvt << " / " <<  _nEvent <<  " events." << std::endl;
  std::cout << "Pt  = " << _passPt  << std::endl;
  std::cout << "EMF = " << _passEMF << std::endl;
}

bool
myFilter::filter(edm::Event& evt, edm::EventSetup const& es) {

  bool result     = false;
  bool filter_Pt  = false;
  bool filter_EMF = false;

  // *********************************************************
  // --- Loop over jets and make a list of all the used towers
  // *********************************************************
  Handle<CaloJetCollection> jets;
  evt.getByLabel( CaloJetAlgorithm, jets );
  for ( CaloJetCollection::const_iterator ijet=jets->begin(); ijet!=jets->end(); ijet++) {
    if (ijet->pt() > 100.)                filter_Pt  = true;
    if (ijet->emEnergyFraction() > 0.05)  filter_EMF = true;
  }

  _nEvent++;  
  if ((filter_Pt) || (filter_EMF)) {
    result = true;
    _acceptedEvt++;
    if (filter_Pt)  _passPt++;
    if (filter_EMF) _passEMF++;
  }

  return result;
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(myFilter);
