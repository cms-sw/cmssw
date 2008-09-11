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
  _rejectedEvt = 0;
  _nEvent      = 0;
}

myFilter::~myFilter() {
}

void myFilter::beginJob(edm::EventSetup const&) {
}

void myFilter::endJob() {

  std::cout << "myFilter: rejected " 
	    << _rejectedEvt << " / " <<  _nEvent <<  " events." << std::endl;

}

bool
myFilter::filter(edm::Event& evt, edm::EventSetup const& es) {

  bool result = false;

  //  int thisEvt = evt.id().event();

  Handle<CaloJetCollection> jets;

  // *********************************************************
  // --- Loop over jets and make a list of all the used towers

  evt.getByLabel( CaloJetAlgorithm, jets );
  for ( CaloJetCollection::const_iterator ijet=jets->begin(); ijet!=jets->end(); ijet++) {
    if (ijet->pt() > 100.) result = true;
    //    cout << "Pt = " << ijet->pt() << endl;
  }

  _nEvent++;
  
  if(!result) _rejectedEvt++;
  return result;
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(myFilter);
