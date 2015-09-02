#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

using namespace edm;
using namespace std;

class LHEfilterOnB : public edm::EDFilter {

  public:

    explicit LHEfilterOnB(const edm::ParameterSet & iConfig);
    ~LHEfilterOnB() {}

  private:

    virtual bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

    edm::InputTag lhesrc_;
};

LHEfilterOnB::LHEfilterOnB(const edm::ParameterSet & iConfig): 
  // LHE collection
  lhesrc_(iConfig.getParameter<InputTag>( "LHEEventProduct" ) )
{

}

bool LHEfilterOnB::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // veto 
  Handle<LHEEventProduct> lheevt;
  iEvent.getByLabel( lhesrc_, lheevt );

  const lhef::HEPEUP hepeup = lheevt->hepeup();
  int nb_lhe = 0;
  cout << "LHE level: ";
	for(int i = 0; i < hepeup.NUP; i++) {
    cout << i << " pdgId= " << hepeup.IDUP[i];
		if (hepeup.ISTUP[i] == 1 && TMath::Abs(hepeup.IDUP[i])==5){
      cout << " --> found a b at LHE level! ";
      nb_lhe++;
      if(nb_lhe>1){
        cout << "found more than 1 b at LHE level, vetoing the event" << endl;
        return false; // skip event
      }
    }
    cout << endl;
	}
  
  return true;
  
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LHEfilterOnB);